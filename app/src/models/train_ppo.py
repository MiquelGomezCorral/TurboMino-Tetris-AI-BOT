import os
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from maikol_utils.print_utils import print_separator, print_warn

from src.config import Configuration
from src.tetris import TetrisConfiguration
from src.models import TurboMinoEncoder, TetrisEnv
from src.models.callbacks import ProgressBarCallback, EntropyAnnealCallback


# ==========================================
# 1. Masking Wrapper Function
# ==========================================
def mask_fn(env: gym.Env):
    return env.unwrapped.valid_action_mask()


def _make_linear_schedule(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


# ==========================================
# 2. Env factories
# ==========================================
def _make_train_env(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    if CONFIG.n_envs > 1:
        return DummyVecEnv([
            lambda: ActionMasker(Monitor(TetrisEnv(CONFIG, T_CONFIG)), mask_fn)
            for _ in range(CONFIG.n_envs)
        ])
    env = TetrisEnv(CONFIG, T_CONFIG)
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)
    return env


def _make_eval_env(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    env = TetrisEnv(CONFIG, T_CONFIG)
    env = ActionMasker(env, mask_fn)
    return env


def _create_fresh_model(CONFIG: Configuration, T_CONFIG: TetrisConfiguration, env, lr_schedule):
    policy_kwargs = dict(
        features_extractor_class=TurboMinoEncoder,
        features_extractor_kwargs=dict(T_CONFIG=T_CONFIG, CONFIG=CONFIG),
        net_arch=dict(pi=CONFIG.net_arch, vf=CONFIG.net_arch),
    )
    return MaskablePPO(
        "MultiInputPolicy", env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        n_steps=CONFIG.n_steps,
        batch_size=CONFIG.batch_size,
        ent_coef=CONFIG.ent_coef,
        clip_range=CONFIG.clip_range,
        gamma=CONFIG.gamma,
        tensorboard_log=CONFIG.log_dir,
        verbose=0,
    )


def _load_model(CONFIG: Configuration, env, lr_schedule):
    print(f" - Resuming training from existing model: {CONFIG.final_model_path}")
    model = MaskablePPO.load(
        CONFIG.final_model_path, env=env, tensorboard_log=CONFIG.log_dir
    )
    model.learning_rate = lr_schedule
    return model


# ==========================================
# 3. Stage runner
# ==========================================
def _run_stage(model, CONFIG: Configuration, T_CONFIG: TetrisConfiguration,
               eval_env, cumulative_target: int, stage_timesteps: int,
               total_timesteps: int, stage_label: str | None = None):
    progress = ProgressBarCallback(
        total_timesteps=stage_timesteps,
        n_steps=CONFIG.n_steps,
        n_envs=CONFIG.n_envs,
    )

    ckpt_dir = CONFIG.checkpoint_dir
    ckpt_prefix = "turbomino_ckpt"
    if stage_label:
        ckpt_dir = os.path.join(ckpt_dir, stage_label)
        ckpt_prefix = f"turbomino_{stage_label}_ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint = CheckpointCallback(
        save_freq=CONFIG.save_freq,
        save_path=ckpt_dir,
        name_prefix=ckpt_prefix,
    )

    validation = TetrisValidationCallback(
        eval_env=eval_env,
        eval_freq=CONFIG.save_freq,
        n_eval_episodes=CONFIG.eval_episodes,
        max_pieces=CONFIG.max_eval_pieces,
    )

    ent_anneal = EntropyAnnealCallback(
        start=CONFIG.ent_coef,
        end=CONFIG.ent_coef_end,
        total_timesteps=total_timesteps,
    )

    try:
        model.learn(
            total_timesteps=cumulative_target,
            callback=[progress, checkpoint, validation, ent_anneal],
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print(f"\n - Stage interrupted by user (step {model.num_timesteps:_}).")


# ==========================================
# 4. Main training entry point
# ==========================================
def train_ppo_turbomino(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    print_separator("Starting PPO training for TurboMino...", sep_type="START")
    CONFIG.print_config()
    T_CONFIG.print_config()

    lr = _make_linear_schedule(CONFIG.learning_rate, CONFIG.lr_end)
    curriculum = CONFIG.curriculum
    if not curriculum:
        curriculum = {T_CONFIG.board_w: CONFIG.total_timesteps}

    # --- Curriculum mode ---
    stages = sorted(curriculum.items())
    total_curriculum = sum(t for _, t in stages)
    if CONFIG.curriculum:
        print_warn(f"Curriculum active — `total_timesteps` ({CONFIG.total_timesteps:_}) "
                   f"ignored; curriculum total is {total_curriculum:_}")

    model = None
    cumulative = 0

    for stage_idx, (board_w, stage_time) in enumerate(stages):
        cumulative += stage_time
        stage_label = f"w{board_w}"

        print(f"\n{'='*70}")
        print(f"  CURRICULUM STAGE {stage_idx+1}/{len(stages)}: board_w={board_w}"
              f"  ({stage_time:_} steps, target: {cumulative:_})")
        print(f"{'='*70}")

        T_CONFIG.board_w = board_w

        env = _make_train_env(CONFIG, T_CONFIG)
        eval_env = _make_eval_env(CONFIG, T_CONFIG)

        if model is None:
            if os.path.exists(CONFIG.final_model_path):
                model = _load_model(CONFIG, env, lr)
            else:
                print(" - Initializing fresh TurboMino model.")
                model = _create_fresh_model(CONFIG, T_CONFIG, env, lr)
        else:
            model.set_env(env)

        _run_stage(model, CONFIG, T_CONFIG, eval_env, cumulative, stage_time, total_curriculum, stage_label)

        stage_path = os.path.join(
            CONFIG.MODELS_PATH,
            f"tetris_turbomino_{CONFIG.exp_name}_{stage_label}.zip",
        )
        model.save(stage_path)
        print(f" - Stage model saved: {stage_path}")

    model.save(CONFIG.final_model_path)
    print(f" - Final model saved to {CONFIG.final_model_path}")


# ==========================================
# 5. Validation callback
# ==========================================
class TetrisValidationCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int = 10000, n_eval_episodes: int = 5,
                 max_pieces: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_pieces = max_pieces
        self.best_mean_score = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            scores = []
            lines = []
            pieces = []

            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                pieces_placed = 0

                while not done and pieces_placed < self.max_pieces:
                    action_masks = self.eval_env.unwrapped.valid_action_mask()
                    action, _ = self.model.predict(
                        obs, action_masks=action_masks, deterministic=True
                    )
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    pieces_placed += 1
                    done = terminated or truncated

                game = self.eval_env.unwrapped.game
                scores.append(game.score_system.score)
                lines.append(game.score_system.lines_cleared_total)
                pieces.append(pieces_placed)

            mean_score = np.mean(scores)
            self.logger.record("val/mean_score", mean_score)
            self.logger.record("val/mean_lines_cleared", np.mean(lines))
            self.logger.record("val/mean_pieces_placed", np.mean(pieces))

            if mean_score > self.best_mean_score:
                self.best_mean_score = mean_score
                save_path = f"{self.model.tensorboard_log}/best_model"
                self.model.save(save_path)
                if self.verbose > 0:
                    print(f"\n - New best validation score: {mean_score:.2f}! Model saved.")

        return True

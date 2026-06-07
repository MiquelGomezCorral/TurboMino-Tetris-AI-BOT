import os
import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

# Import your environment, configs, and neural network
from src.config import Configuration
from src.tetris import TetrisConfiguration
from src.models import TurboMino, TetrisEnv
from src.models.callbacks import ProgressBarCallback, EntropyAnnealCallback

# ==========================================
# 1. Masking Wrapper Function
# ==========================================
def mask_fn(env: gym.Env):
    """
    sb3-contrib ActionMasker looks for a function that returns the boolean mask.
    In our TetrisEnv, we exposed this via the 'placement_mask' in the dict, 
    but we can also just call the underlying env method.
    """
    # If the env is wrapped in standard SB3 wrappers, we need to access the un-wrapped env
    return env.unwrapped.valid_action_mask()


def _make_linear_schedule(start: float, end: float):
    """SB3-compatible linear LR schedule.  progress_remaining: 1.0 → 0.0."""
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


# ==========================================
# 2. Main Training Loop
# ==========================================
def train_turbomino(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    # --- Environment Setup ---
    if CONFIG.n_envs > 1:
        env = DummyVecEnv([
            lambda: ActionMasker(Monitor(TetrisEnv(CONFIG, T_CONFIG)), mask_fn)
            for _ in range(CONFIG.n_envs)
        ])
    else:
        env = TetrisEnv(CONFIG, T_CONFIG)
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)

    eval_env = TetrisEnv(CONFIG, T_CONFIG)
    eval_env = ActionMasker(eval_env, mask_fn)

    # --- Learning rate schedule ---
    lr = _make_linear_schedule(CONFIG.learning_rate, CONFIG.lr_end)

    # --- Initialization / Resuming ---
    if os.path.exists(CONFIG.final_model_path):
        print(f"[*] Resuming training from existing model: {CONFIG.final_model_path}")
        model = MaskablePPO.load(
            CONFIG.final_model_path, 
            env=env, 
            tensorboard_log=CONFIG.log_dir
        )
        model.learning_rate = lr
    else:
        print("[*] Initializing fresh TurboMino model.")
        
        policy_kwargs = dict(
            features_extractor_class=TurboMino,
            features_extractor_kwargs=dict(
                T_CONFIG=T_CONFIG,
                CONFIG=CONFIG
            ),
            net_arch=dict(pi=CONFIG.net_arch, vf=CONFIG.net_arch)
        )

        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=lr,
            n_steps=CONFIG.n_steps,
            batch_size=CONFIG.batch_size,
            ent_coef=CONFIG.ent_coef,
            clip_range=CONFIG.clip_range,
            gamma=CONFIG.gamma,
            tensorboard_log=CONFIG.log_dir,
            verbose=0,
        )

    # --- Callbacks ---
    progress_callback = ProgressBarCallback(
        total_timesteps=CONFIG.total_timesteps,
        n_steps=CONFIG.n_steps,
        n_envs=CONFIG.n_envs,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=CONFIG.save_freq,
        save_path=CONFIG.checkpoint_dir,
        name_prefix="turbomino_ckpt"
    )

    validation_callback = TetrisValidationCallback(
        eval_env=eval_env,
        eval_freq=CONFIG.save_freq,
        n_eval_episodes=CONFIG.eval_episodes,
        max_pieces=CONFIG.max_eval_pieces
    )

    ent_coef_callback = EntropyAnnealCallback(
        start=CONFIG.ent_coef,
        end=CONFIG.ent_coef_end,
        total_timesteps=CONFIG.total_timesteps,
    )

    # --- Execution ---
    print("[*] Starting training loop...")
    try:
        model.learn(
            total_timesteps=CONFIG.total_timesteps, 
            callback=[progress_callback, checkpoint_callback, validation_callback, ent_coef_callback],
            reset_num_timesteps=False 
        )
    except KeyboardInterrupt:
        print("\n[*] Training interrupted by user. Saving current state...")
    finally:
        model.save(CONFIG.final_model_path)
        print(f"[*] Model saved to {CONFIG.final_model_path}")


class TetrisValidationCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int = 10000, n_eval_episodes: int = 5, max_pieces: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.max_pieces = max_pieces
        self.best_mean_score = -np.inf

    def _on_step(self) -> bool:
        # Trigger evaluation every eval_freq steps
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            scores = []
            lines = []
            pieces = []

            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                pieces_placed = 0
                
                # Play until death OR the 100 piece limit
                while not done and pieces_placed < self.max_pieces:
                    # 1. Manually fetch the mask from the environment
                    action_masks = self.eval_env.unwrapped.valid_action_mask()
                    
                    # 2. Predict deterministically (No random exploration during validation)
                    action, _ = self.model.predict(
                        obs, 
                        action_masks=action_masks, 
                        deterministic=True
                    )
                    
                    # 3. Step the environment
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    pieces_placed += 1
                    done = terminated or truncated

                # Extract exact game metrics from the underlying Tetris engine
                game = self.eval_env.unwrapped.game
                scores.append(game.score_system.score)
                lines.append(game.score_system.lines_cleared_total)
                pieces.append(pieces_placed)

            mean_score = np.mean(scores)
            
            # Log metrics to TensorBoard
            self.logger.record("val/mean_score", mean_score)
            self.logger.record("val/mean_lines_cleared", np.mean(lines))
            self.logger.record("val/mean_pieces_placed", np.mean(pieces))
            
            # Save the best model
            if mean_score > self.best_mean_score:
                self.best_mean_score = mean_score
                save_path = f"{self.model.tensorboard_log}/best_model"
                self.model.save(save_path)
                if self.verbose > 0:
                    print(f"\n[*] New best validation score: {mean_score:.2f}! Model saved.")

        return True
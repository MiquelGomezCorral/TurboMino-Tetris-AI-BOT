import os
from stable_baselines3.common.callbacks import CheckpointCallback

from maikol_utils.print_utils import print_separator, print_warn

from src.config import Configuration
from src.tetris import TetrisConfiguration
from .callbacks import ProgressBarCallback, EntropyAnnealCallback, TetrisValidationCallback
from .utils import load_model, create_fresh_model
from .gym_env import make_eval_env, make_train_env

# ==========================================
# 1. Masking Wrapper Function
# ==========================================
def _make_linear_schedule(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


# ==========================================
# 2. Stage runner
# ==========================================
def _run_stage(
    model, CONFIG: Configuration, T_CONFIG: TetrisConfiguration,
    eval_env, stage_timesteps: int,
    total_timesteps: int, stage_label: str | None = None,
    use_curriculum_gate: bool = False,
):
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
        learned_ratio=CONFIG.curriculum_learned_ratio if use_curriculum_gate else None,
        min_reward=CONFIG.curriculum_min_eval_reward,
    )

    ent_anneal = EntropyAnnealCallback(
        start=CONFIG.ent_coef,
        end=CONFIG.ent_coef_end,
        total_timesteps=total_timesteps,
    )

    callbacks = [progress, checkpoint, validation, ent_anneal]

    try:
        model.learn(
            total_timesteps=stage_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print(f"\n - Stage interrupted by user (step {model.num_timesteps:_}).")
    return not use_curriculum_gate or validation.learned


# ==========================================
# 3. Main training entry point
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

        env = make_train_env(CONFIG, T_CONFIG)
        eval_env = make_eval_env(CONFIG, T_CONFIG)

        if model is None:
            if os.path.exists(CONFIG.final_model_path):
                model = load_model(CONFIG, T_CONFIG, env, lr)
            else:
                print(" - Initializing fresh TurboMino model.")
                model = create_fresh_model(CONFIG, T_CONFIG, env, lr)
        else:
            model.set_env(env)

        learned = _run_stage(
            model, CONFIG, T_CONFIG, eval_env, stage_time, total_curriculum, stage_label,
            use_curriculum_gate=bool(CONFIG.curriculum),
        )

        stage_path = os.path.join(
            CONFIG.MODELS_PATH,
            f"tetris_turbomino_{CONFIG.exp_name}_{stage_label}.zip",
        )
        model.save(stage_path)
        print(f" - Stage model saved: {stage_path}")
        if CONFIG.curriculum and not learned:
            print_warn(f"Curriculum stopped: {stage_label} did not pass the learning gate.")
            break

    model.save(CONFIG.final_model_path)
    print(f" - Final model saved to {CONFIG.final_model_path}")

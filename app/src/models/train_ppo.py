import os
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import LinearSchedule

from maikol_utils.print_utils import print_separator, print_warn

from src.config import Configuration
from src.tetris import TetrisConfiguration
from .callbacks import ProgressBarCallback, EntropyAnnealCallback, TetrisValidationCallback
from .utils import load_model, create_fresh_model
from .gym_env import make_eval_env, make_train_env
from .test import test_on_game

# ==========================================
# 1. Stage runner
# ==========================================
def _run_stage(
    model, CONFIG: Configuration, T_CONFIG: TetrisConfiguration,
    eval_env, stage_timesteps: int,
    total_timesteps: int, stage_label: str | None = None,
    use_curriculum_gate: bool = False,
):
    progress = ProgressBarCallback(
        total_timesteps=stage_timesteps,
        rollout_steps=CONFIG.rollout_steps(),
        n_envs=CONFIG.n_envs,
    )

    ckpt_dir = CONFIG.checkpoint_dir
    ckpt_prefix = "turbomino_ckpt"
    if stage_label:
        ckpt_dir = os.path.join(ckpt_dir, stage_label)
        ckpt_prefix = f"turbomino_{stage_label}_ckpt"
    os.makedirs(ckpt_dir, exist_ok=True)

    stage_end = model.num_timesteps + stage_timesteps
    stage_end_lr = CONFIG.lr_end + (CONFIG.learning_rate - CONFIG.lr_end) * max(
        0.0, 1.0 - stage_end / total_timesteps,
    )
    model.learning_rate = LinearSchedule(CONFIG.learning_rate, stage_end_lr, 1.0)
    model._setup_lr_schedule()

    checkpoint = CheckpointCallback(
        save_freq=CONFIG.eval_steps(),
        save_path=ckpt_dir,
        name_prefix=ckpt_prefix,
    )

    validation = TetrisValidationCallback(
        eval_env=eval_env,
        eval_freq=CONFIG.eval_steps(),
        n_eval_episodes=CONFIG.eval_episodes,
        max_pieces=CONFIG.max_eval_pieces,
        learned_ratio=CONFIG.curriculum_learned_ratio if use_curriculum_gate else None,
        min_score=CONFIG.curriculum_min_eval_score,
        eval_seed=CONFIG.eval_seed,
        best_model_path=os.path.join(ckpt_dir, "best_model.zip"),
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
    return not use_curriculum_gate or validation.learned, validation.saved_best_model_path


# ==========================================
# 2. Main training entry point
# ==========================================
def train_ppo_turbomino(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    print_separator("Starting PPO training for TurboMino...", sep_type="START")
    CONFIG.print_config()
    T_CONFIG.print_config()

    lr = CONFIG.learning_rate
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
            if CONFIG.resume_model_path:
                if CONFIG.resume_model_path.endswith(".ckpt"):
                    raise ValueError("--resume_model_path must point to a MaskablePPO .zip checkpoint")
                model = load_model(
                    CONFIG, T_CONFIG, env, lr, model_path=CONFIG.resume_model_path,
                )
            else:
                print(" - Initializing fresh TurboMino model.")
                model = create_fresh_model(CONFIG, T_CONFIG, env, lr)
        else:
            model.set_env(env)

        learned, best_model_path = _run_stage(
            model, CONFIG, T_CONFIG, eval_env, stage_time, total_curriculum, stage_label,
            use_curriculum_gate=bool(CONFIG.curriculum),
        )

        if best_model_path and os.path.exists(best_model_path):
            completed_timesteps = model.num_timesteps
            model = load_model(CONFIG, T_CONFIG, env, lr, model_path=best_model_path)
            model.num_timesteps = completed_timesteps

        stage_path = os.path.join(
            CONFIG.MODELS_PATH,
            f"tetris_turbomino_{CONFIG.exp_name}_{stage_label}.zip",
        )
        model.save(stage_path)
        print(f" - Stage model saved: {stage_path}")
        env.close()
        eval_env.close()
        if CONFIG.curriculum and not learned:
            print_warn(f"Curriculum stopped: {stage_label} did not pass the learning gate.")
            break

    model.save(CONFIG.final_model_path)
    print(f" - Final model saved to {CONFIG.final_model_path}")

    final_eval_env = make_eval_env(CONFIG, T_CONFIG)
    try:
        _, scores, _, pieces, _, _ = test_on_game(
            n_eval_episodes=CONFIG.eval_episodes,
            max_pieces=CONFIG.max_eval_pieces,
            eval_env=final_eval_env,
            model=model,
            seed=CONFIG.eval_seed + CONFIG.eval_episodes,
        )
        print(
            f" - Final held-out evaluation: score={sum(scores) / len(scores):.1f}, "
            f"pieces=min:{min(pieces)}, avg:{sum(pieces) / len(pieces):.1f}, max:{max(pieces)}"
        )
    finally:
        final_eval_env.close()

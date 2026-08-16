import os

from stable_baselines3.common.callbacks import CheckpointCallback

from maikol_utils.print_utils import print_separator, print_warn
from maikol_utils.file_utils import make_dirs

from src.config.checkpoints import (
    BEST_MODEL_FILENAME,
    DEFAULT_CHECKPOINT_PREFIX,
    STAGE_CHECKPOINT_PREFIX,
    STAGE_MODEL_FILENAME,
)
from src.config import Configuration
from src.tetris import TetrisConfiguration
from .callbacks import (
    PPOProgressCallback,
    TetrisValidationCallback,
)
from .utils import load_model, create_fresh_model
from .gym_env import make_eval_env, make_train_env
from .test import test_on_game
from .train_ppo_utils import (
    _load_resume_state,
    _save_resume_state,
    _stage_index_from_checkpoint,
    _warn_resume_config_differences,
)


class ResumeCheckpointCallback(CheckpointCallback):
    def __init__(
        self,
        *args,
        CONFIG: Configuration,
        T_CONFIG: TetrisConfiguration,
        stage_index: int,
        stage_start_global_steps: int,
        stage_target_steps: int,
        use_curriculum_gate: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.CONFIG = CONFIG
        self.T_CONFIG = T_CONFIG
        self.stage_index = stage_index
        self.stage_start_global_steps = stage_start_global_steps
        self.stage_target_steps = stage_target_steps
        self.use_curriculum_gate = use_curriculum_gate
        self.validation = None

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps.zip",
            )
            stage_completed_steps = max(
                0, self.num_timesteps - self.stage_start_global_steps
            )
            _save_resume_state(
                checkpoint_path,
                self.CONFIG,
                self.T_CONFIG,
                self.stage_index,
                self.stage_start_global_steps,
                self.stage_target_steps,
                stage_completed_steps,
                self.validation.learned
                if self.use_curriculum_gate and self.validation
                else (
                    not self.use_curriculum_gate
                    and stage_completed_steps >= self.stage_target_steps
                ),
            )
        return result


# ==========================================
# 1. Stage runner
# ==========================================
def _run_stage(
    model, CONFIG: Configuration, T_CONFIG: TetrisConfiguration,
    eval_env, stage_timesteps: int,
    stage_index: int, stage_start_global_steps: int,
    stage_target_steps: int, stage_label: str | None = None,
    use_curriculum_gate: bool = False,
):
    ckpt_dir = CONFIG.checkpoint_dir
    ckpt_prefix = DEFAULT_CHECKPOINT_PREFIX
    if stage_label:
        ckpt_dir = os.path.join(ckpt_dir, stage_label)
        ckpt_prefix = STAGE_CHECKPOINT_PREFIX.format(stage_label=stage_label)

    make_dirs(ckpt_dir)

    checkpoint = ResumeCheckpointCallback(
        save_freq=CONFIG.eval_steps(),
        save_path=ckpt_dir,
        name_prefix=ckpt_prefix,
        CONFIG=CONFIG,
        T_CONFIG=T_CONFIG,
        stage_index=stage_index,
        stage_start_global_steps=stage_start_global_steps,
        stage_target_steps=stage_target_steps,
        use_curriculum_gate=use_curriculum_gate,
    )

    def save_best_model_state(model_path, num_timesteps, learned):
        stage_completed_steps = max(0, num_timesteps - stage_start_global_steps)
        _save_resume_state(
            model_path,
            CONFIG,
            T_CONFIG,
            stage_index,
            stage_start_global_steps,
            stage_target_steps,
            stage_completed_steps,
            learned if use_curriculum_gate else stage_completed_steps >= stage_target_steps,
        )

    CONFIG.best_model_path = os.path.join(ckpt_dir, BEST_MODEL_FILENAME)
    validation = TetrisValidationCallback(
        eval_env=eval_env,
        eval_freq=CONFIG.eval_steps(),
        n_eval_episodes=CONFIG.eval_episodes,
        max_pieces=CONFIG.max_eval_pieces,
        learned_ratio=CONFIG.curriculum_learned_ratio if use_curriculum_gate else None,
        min_reward=CONFIG.curriculum_min_eval_reward,
        eval_seed=CONFIG.eval_seed,
        best_model_path=CONFIG.best_model_path,
        model_path_template=os.path.join(
            ckpt_dir, f"{ckpt_prefix}_{{num_timesteps}}_steps.zip"
        ),
        on_best_model=save_best_model_state,
    )

    checkpoint.validation = validation
    callbacks = [validation, checkpoint, PPOProgressCallback(stage_timesteps)]

    try:
        model.learn(
            total_timesteps=stage_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print(f"\n - Stage interrupted by user (step {model.num_timesteps:_}).")
    stage_completed_steps = max(0, model.num_timesteps - stage_start_global_steps)
    stage_complete = (
        validation.learned
        if use_curriculum_gate
        else stage_completed_steps >= stage_target_steps
    )
    return stage_complete


# ==========================================
# 2. Main training entry point
# ==========================================
def train_ppo_turbomino(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    print_separator("Starting PPO training for TurboMino...", sep_type="START")
    _prepare_training(CONFIG, T_CONFIG)
    stages = _build_curriculum(CONFIG, T_CONFIG)
    resume_state, resume_stage_index = _load_resume_context(CONFIG, T_CONFIG, stages)
    model, last_stage = _run_curriculum(
        CONFIG,
        T_CONFIG,
        stages,
        resume_state,
        resume_stage_index,
    )
    _save_final_model(model, CONFIG, T_CONFIG, last_stage)
    _run_final_evaluation(CONFIG, T_CONFIG)


def _prepare_training(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    make_dirs([CONFIG.MODELS_PATH, CONFIG.log_dir, CONFIG.checkpoint_dir])
    CONFIG.print_config()
    T_CONFIG.print_config()


def _build_curriculum(CONFIG: Configuration, T_CONFIG: TetrisConfiguration):
    curriculum = CONFIG.curriculum or {T_CONFIG.board_w: CONFIG.total_timesteps}
    stages = list(curriculum.items())
    if CONFIG.curriculum:
        total_curriculum = sum(timesteps for _, timesteps in stages)
        print_warn(
            f"Curriculum active — `total_timesteps` ({CONFIG.total_timesteps:_}) "
            f"ignored; curriculum total is {total_curriculum:_}"
        )
    return stages


def _load_resume_context(CONFIG, T_CONFIG, stages):
    if not CONFIG.resume_model_path:
        return None, 0

    if os.path.splitext(CONFIG.resume_model_path)[1].lower() == ".ckpt":
        return None, 0

    resume_state = _load_resume_state(CONFIG.resume_model_path)
    if resume_state:
        _warn_resume_config_differences(
            resume_state.get("config", {}),
            CONFIG,
            resume_state.get("tetris", {}),
            T_CONFIG,
        )
        saved_curriculum = resume_state.get("curriculum", {})
        saved_width = saved_curriculum.get("board_width")
        resume_stage_index = next(
            (
                index
                for index, (board_w, _) in enumerate(stages)
                if board_w == saved_width
            ),
            saved_curriculum.get("stage_index", 0),
        )
    else:
        resume_stage_index = _stage_index_from_checkpoint(
            CONFIG.resume_model_path, stages
        )

    resume_stage_index = min(resume_stage_index, len(stages) - 1)
    saved_width = (resume_state or {}).get("curriculum", {}).get("board_width")
    current_width = stages[resume_stage_index][0]
    if saved_width is not None and saved_width != current_width:
        print_warn(
            f"Resume curriculum width differs: "
            f"checkpoint={saved_width!r}, current={current_width!r}"
        )
    return resume_state, resume_stage_index


def _run_curriculum(CONFIG, T_CONFIG, stages, resume_state, resume_stage_index):
    model = None
    cumulative = 0
    last_stage = None

    for stage_idx, (board_w, stage_time) in enumerate(stages):
        cumulative += stage_time
        if CONFIG.resume_model_path and stage_idx < resume_stage_index:
            continue

        model, last_stage, stage_complete = _run_curriculum_stage(
            model,
            CONFIG,
            T_CONFIG,
            stage_idx,
            board_w,
            stage_time,
            cumulative,
            len(stages),
            resume_state,
            resume_stage_index,
        )
        if CONFIG.curriculum and not stage_complete:
            print_warn(
                f"Curriculum stopped: w{board_w} did not pass the learning gate."
            )
            break

    return model, last_stage


def _run_curriculum_stage(
    model,
    CONFIG,
    T_CONFIG,
    stage_idx,
    board_w,
    stage_time,
    cumulative,
    stage_count,
    resume_state,
    resume_stage_index,
):
    stage_label = f"w{board_w}"
    stage_start_global_steps = cumulative - stage_time

    print(f"\n{'=' * 70}")
    print(
        f"  CURRICULUM STAGE {stage_idx + 1}/{stage_count}: board_w={board_w}"
        f"  ({stage_time:_} steps, target: {cumulative:_})"
    )
    print(f"{'=' * 70}")

    T_CONFIG.board_w = board_w
    env = None
    eval_env = None
    try:
        env = make_train_env(CONFIG, T_CONFIG)
        eval_env = make_eval_env(CONFIG, T_CONFIG)
        model = _prepare_model(model, CONFIG, T_CONFIG, env)
        stage_start_global_steps, stage_completed_steps = _stage_progress(
            model,
            CONFIG,
            stage_idx,
            board_w,
            stage_time,
            stage_start_global_steps,
            resume_state,
            resume_stage_index,
        )
        remaining_steps = max(0, stage_time - stage_completed_steps)
        if remaining_steps:
            stage_complete = _run_stage(
                model,
                CONFIG,
                T_CONFIG,
                eval_env,
                remaining_steps,
                stage_idx,
                stage_start_global_steps,
                stage_time,
                stage_label,
                use_curriculum_gate=bool(CONFIG.curriculum),
            )
        else:
            saved_curriculum = (resume_state or {}).get("curriculum", {})
            stage_complete = (
                saved_curriculum.get("stage_complete", False)
                if CONFIG.curriculum
                else True
            )

        stage_completed_steps = max(
            stage_completed_steps,
            model.num_timesteps - stage_start_global_steps,
        )
        stage_completed_steps = min(stage_time, stage_completed_steps)
        last_stage = _save_stage_model(
            model,
            CONFIG,
            T_CONFIG,
            stage_idx,
            stage_label,
            stage_start_global_steps,
            stage_time,
            stage_completed_steps,
            stage_complete,
        )
        return model, last_stage, stage_complete
    finally:
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()


def _prepare_model(model, CONFIG, T_CONFIG, env):
    if model is not None:
        model.set_env(env)
        return model

    if CONFIG.resume_model_path:
        model = load_model(
            CONFIG,
            T_CONFIG,
            env,
            CONFIG.learning_rate,
            model_path=CONFIG.resume_model_path,
        )
    else:
        print(" - Initializing fresh TurboMino model.")
        model = create_fresh_model(CONFIG, T_CONFIG, env, CONFIG.learning_rate)
    model.policy.features_extractor.print_parameters()
    return model


def _stage_progress(
    model,
    CONFIG,
    stage_idx,
    board_w,
    stage_time,
    stage_start_global_steps,
    resume_state,
    resume_stage_index,
):
    is_ppo_resume = (
        CONFIG.resume_model_path
        and os.path.splitext(CONFIG.resume_model_path)[1].lower() == ".zip"
    )
    if is_ppo_resume and stage_idx == resume_stage_index:
        saved_curriculum = (resume_state or {}).get("curriculum", {})
        if saved_curriculum.get("board_width") == board_w:
            stage_start_global_steps = saved_curriculum.get(
                "stage_start_global_steps", stage_start_global_steps
            )
        stage_completed_steps = (
            stage_time
            if saved_curriculum.get("stage_complete")
            else max(0, model.num_timesteps - stage_start_global_steps)
        )
        stage_completed_steps = min(stage_time, stage_completed_steps)
        print(
            f" - Resuming stage w{board_w}: "
            f"{stage_completed_steps:_}/{stage_time:_} steps already done."
        )
        return stage_start_global_steps, stage_completed_steps

    if model.num_timesteps:
        stage_start_global_steps = model.num_timesteps
    return stage_start_global_steps, 0


def _save_stage_model(
    model,
    CONFIG,
    T_CONFIG,
    stage_idx,
    stage_label,
    stage_start_global_steps,
    stage_time,
    stage_completed_steps,
    stage_complete,
):
    stage_path = os.path.join(
        CONFIG.MODELS_PATH,
        STAGE_MODEL_FILENAME.format(
            exp_name=CONFIG.exp_name,
            stage_label=stage_label,
        ),
    )
    model.save(stage_path)
    _save_resume_state(
        stage_path,
        CONFIG,
        T_CONFIG,
        stage_idx,
        stage_start_global_steps,
        stage_time,
        stage_completed_steps,
        stage_complete,
    )
    print(f" - Stage model saved: {stage_path}")
    return (
        stage_idx,
        stage_start_global_steps,
        stage_time,
        stage_completed_steps,
        stage_complete,
    )


def _save_final_model(model, CONFIG, T_CONFIG, last_stage):
    model.save(CONFIG.final_model_path)
    if last_stage:
        _save_resume_state(CONFIG.final_model_path, CONFIG, T_CONFIG, *last_stage)
    print(f" - Final model saved to {CONFIG.final_model_path}")


def _run_final_evaluation(CONFIG, T_CONFIG):
    if not CONFIG.run_final_eval:
        return

    _, scores, _, pieces, _, _ = test_on_game(
        CONFIG=CONFIG,
        T_CONFIG=T_CONFIG,
        eval_seed=(
            CONFIG.eval_seed + CONFIG.eval_episodes
            if CONFIG.eval_seed is not None
            else None
        ),
    )
    print(
        f" - Final held-out evaluation: score={sum(scores) / len(scores):.1f}, "
        f"pieces=min:{min(pieces)}, avg:{sum(pieces) / len(pieces):.1f}, max:{max(pieces)}"
    )

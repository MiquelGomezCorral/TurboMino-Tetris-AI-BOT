import os
import re
from dataclasses import asdict

import numpy as np
import yaml
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import LinearSchedule

from maikol_utils.print_utils import print_separator, print_warn
from maikol_utils.file_utils import make_dirs

from src.config import Configuration
from src.tetris import TetrisConfiguration
from .callbacks import ProgressBarCallback, EntropyAnnealCallback, TetrisValidationCallback
from .utils import load_model, create_fresh_model
from .gym_env import make_eval_env, make_train_env
from .test import test_on_game


RESUME_STATE_SUFFIX = ".resume.yaml"
IGNORED_RESUME_CONFIG_FIELDS = {
    "config",
    "model_path",
    "resume_model_path",
    "DATA_PATH",
    "MODELS_PATH",
    "LOGS_PATH",
    "CONFIGS_PATH",
    "raw_dataset_path",
    "processed_dataset_path",
    "tetrio_train",
    "tetrio_test",
    "tetrio_val",
    "pretrain_model_path",
    "checkpoint_dir",
    "log_dir",
    "final_model_path",
}


def _resume_state_path(model_path: str) -> str:
    return f"{model_path}{RESUME_STATE_SUFFIX}"


def _yaml_safe(value):
    if isinstance(value, dict):
        return {_yaml_safe(key): _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _save_resume_state(
    model_path: str,
    CONFIG: Configuration,
    T_CONFIG: TetrisConfiguration,
    stage_index: int,
    stage_start_global_steps: int,
    stage_target_steps: int,
    stage_completed_steps: int,
    stage_complete: bool,
):
    state = {
        "version": 1,
        "config": asdict(CONFIG),
        "tetris": {
            "board_w": T_CONFIG.board_w,
            "board_h": T_CONFIG.board_h,
            "vanish_zone": T_CONFIG.vanish_zone,
            "max_pieces_in_view": T_CONFIG.max_pieces_in_view,
            "num_piece_categories": T_CONFIG.num_piece_categories,
        },
        "curriculum": {
            "stage_index": stage_index,
            "board_width": T_CONFIG.board_w,
            "stage_start_global_steps": stage_start_global_steps,
            "stage_target_steps": stage_target_steps,
            "stage_completed_steps": stage_completed_steps,
            "global_steps": stage_start_global_steps + stage_completed_steps,
            "stage_complete": stage_complete,
        },
    }
    with open(_resume_state_path(model_path), "w", encoding="utf-8") as file:
        yaml.safe_dump(_yaml_safe(state), file, sort_keys=False)


def _load_resume_state(model_path: str):
    state_path = _resume_state_path(model_path)
    if not os.path.exists(state_path):
        print_warn(f"No resume metadata found next to {model_path}; inferring the curriculum stage.")
        return None
    with open(state_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _warn_resume_config_differences(
    saved_config: dict,
    CONFIG: Configuration,
    saved_tetris: dict,
    T_CONFIG: TetrisConfiguration,
):
    current_config = asdict(CONFIG)
    for key in sorted(set(saved_config) | set(current_config)):
        if key in IGNORED_RESUME_CONFIG_FIELDS:
            continue
        saved_value = saved_config.get(key)
        current_value = current_config.get(key)
        if saved_value != current_value:
            print_warn(
                f"Resume config differs for {key}: "
                f"checkpoint={saved_value!r}, current={current_value!r}"
            )
    current_tetris = {
        key: getattr(T_CONFIG, key)
        for key in ("board_h", "vanish_zone", "max_pieces_in_view", "num_piece_categories")
    }
    for key in current_tetris:
        if key in saved_tetris and saved_tetris[key] != current_tetris[key]:
            print_warn(
                f"Resume Tetris config differs for {key}: "
                f"checkpoint={saved_tetris.get(key)!r}, current={current_tetris[key]!r}"
            )


def _stage_index_from_checkpoint(model_path: str, stages: list[tuple[int, int]]) -> int:
    match = re.search(r"(?:^|[/\\_])w(-?\d+)(?=[/\\_.]|$)", model_path)
    if match:
        board_width = int(match.group(1))
        for index, (stage_width, _) in enumerate(stages):
            if stage_width == board_width:
                return index
    return 0


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
    total_timesteps: int, stage_index: int, stage_start_global_steps: int,
    stage_target_steps: int, stage_label: str | None = None,
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

    make_dirs(ckpt_dir)

    stage_end = model.num_timesteps + stage_timesteps
    stage_end_lr = CONFIG.lr_end + (CONFIG.learning_rate - CONFIG.lr_end) * max(
        0.0, 1.0 - stage_end / total_timesteps,
    )
    model.learning_rate = LinearSchedule(CONFIG.learning_rate, stage_end_lr, 1.0)
    model._setup_lr_schedule()

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

    validation = TetrisValidationCallback(
        eval_env=eval_env,
        eval_freq=CONFIG.eval_steps(),
        n_eval_episodes=CONFIG.eval_episodes,
        max_pieces=CONFIG.max_eval_pieces,
        learned_ratio=CONFIG.curriculum_learned_ratio if use_curriculum_gate else None,
        min_score=CONFIG.curriculum_min_eval_score,
        eval_seed=CONFIG.eval_seed,
        best_model_path=os.path.join(ckpt_dir, "best_model.zip"),
        model_path_template=os.path.join(
            ckpt_dir, f"{ckpt_prefix}_{{num_timesteps}}_steps.zip"
        ),
        on_best_model=save_best_model_state,
    )

    ent_anneal = EntropyAnnealCallback(
        start=CONFIG.ent_coef,
        end=CONFIG.ent_coef_end,
        total_timesteps=total_timesteps,
    )

    checkpoint.validation = validation
    callbacks = [progress, checkpoint, validation, ent_anneal]

    try:
        model.learn(
            total_timesteps=stage_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print(f"\n - Stage interrupted by user (step {model.num_timesteps:_}).")
    stage_completed_steps = max(0, model.num_timesteps - stage_start_global_steps)
    stage_complete = (
        validation.learned
        if use_curriculum_gate
        else stage_completed_steps >= stage_target_steps
    )
    return stage_complete, validation.saved_best_model_path


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
    stages = list(curriculum.items())
    total_curriculum = sum(t for _, t in stages)
    if CONFIG.curriculum:
        print_warn(f"Curriculum active — `total_timesteps` ({CONFIG.total_timesteps:_}) "
                   f"ignored; curriculum total is {total_curriculum:_}")

    resume_state = None
    resume_stage_index = 0
    if CONFIG.resume_model_path:
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

    model = None
    cumulative = 0
    last_stage = None

    for stage_idx, (board_w, stage_time) in enumerate(stages):
        cumulative += stage_time
        if CONFIG.resume_model_path and stage_idx < resume_stage_index:
            continue

        stage_label = f"w{board_w}"
        stage_start_global_steps = cumulative - stage_time

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

        stage_completed_steps = 0
        if CONFIG.resume_model_path and stage_idx == resume_stage_index:
            saved_curriculum = (resume_state or {}).get("curriculum", {})
            if saved_curriculum.get("stage_complete"):
                stage_completed_steps = stage_time
            elif (
                saved_curriculum.get("board_width") == board_w
                and saved_curriculum.get("stage_start_global_steps") == stage_start_global_steps
            ):
                stage_completed_steps = saved_curriculum.get("stage_completed_steps", 0)
            else:
                stage_completed_steps = max(
                    0, model.num_timesteps - stage_start_global_steps
                )
            stage_completed_steps = min(stage_time, stage_completed_steps)
            print(
                f" - Resuming stage w{board_w}: "
                f"{stage_completed_steps:_}/{stage_time:_} steps already done."
            )

        remaining_steps = max(0, stage_time - stage_completed_steps)
        if remaining_steps:
            stage_complete, best_model_path = _run_stage(
                model,
                CONFIG,
                T_CONFIG,
                eval_env,
                remaining_steps,
                total_curriculum,
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
            best_model_path = None

        if best_model_path and os.path.exists(best_model_path):
            completed_timesteps = model.num_timesteps
            model = load_model(CONFIG, T_CONFIG, env, lr, model_path=best_model_path)
            model.num_timesteps = completed_timesteps

        stage_completed_steps = max(
            stage_completed_steps,
            model.num_timesteps - stage_start_global_steps,
        )
        stage_completed_steps = min(stage_time, stage_completed_steps)

        stage_path = os.path.join(
            CONFIG.MODELS_PATH,
            f"tetris_turbomino_{CONFIG.exp_name}_{stage_label}.zip",
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
        env.close()
        eval_env.close()
        last_stage = (
            stage_idx,
            stage_start_global_steps,
            stage_time,
            stage_completed_steps,
            stage_complete,
        )
        if CONFIG.curriculum and not stage_complete:
            print_warn(f"Curriculum stopped: {stage_label} did not pass the learning gate.")
            break

    model.save(CONFIG.final_model_path)
    if last_stage:
        _save_resume_state(
            CONFIG.final_model_path,
            CONFIG,
            T_CONFIG,
            *last_stage,
        )
    print(f" - Final model saved to {CONFIG.final_model_path}")

    if CONFIG.run_final_eval:
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

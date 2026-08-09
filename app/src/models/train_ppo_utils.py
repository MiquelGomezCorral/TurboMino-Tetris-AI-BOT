"""Stateless resume and checkpoint helpers for PPO training."""

import os
import re
from dataclasses import asdict

import numpy as np
import yaml

from maikol_utils.print_utils import print_warn

from src.config import Configuration
from src.config.checkpoints import (
    IGNORED_RESUME_CONFIG_FIELDS,
    RESUME_STATE_SUFFIX,
    RESUME_STATE_VERSION,
)
from src.tetris import TetrisConfiguration


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
        "version": RESUME_STATE_VERSION,
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

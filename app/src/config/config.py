"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
from dataclasses import dataclass, field

from maikol_utils.file_utils import make_dirs
import yaml

@dataclass 
class Configuration:
    """Configuration class for the project."""
    # ===================================================================
    #                       PATHS
    # ===================================================================
    exp_name: str = "base_name"


    DATA_PATH: str = os.path.join("..", "data")
    MODELS_PATH: str = os.path.join("..", "models")
    LOGS_PATH: str = os.path.join("..", "logs")
    CONFIGS_PATH: str = os.path.join("..", "configs")
    config: str = None

    raw_dataset_path: str = os.path.join(DATA_PATH, "data.csv")

    checkpoint_dir: str = os.path.join(MODELS_PATH, "checkpoints")
    log_dir: str = os.path.join(LOGS_PATH, f"tensorboard_{exp_name}")
    final_model_path: str = os.path.join(MODELS_PATH, f"tetris_turbomino_{exp_name}.zip")
    # ===================================================================
    #                       PARAMETER
    # ===================================================================

    seed:     int = 42
    gym_id:          str = None
    total_timesteps: int = 25_000

    max_placements: int = 128
    d_model: int = 128
    n_heads: int = 4
    head_hidden: int = 256
    n_piece_layers: int = 2
    max_board_size_w: int = 10
    max_board_size_h: int = 20



    net_arch: list[int] = field(default_factory=lambda: [128, 128])
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 256
    ent_coef: float = 0.015
    gamma: float = 0.99
    verbose: int = 1

    save_freq: int = 10000
    eval_episodes: int = 10
    max_eval_pieces: int = 100

    total_timesteps: int = 5_000_000


    def __post_init__(self):
        # Basic setup: create folders
        make_dirs([
            self.DATA_PATH, 
            self.MODELS_PATH, 
            self.LOGS_PATH,
            self.checkpoint_dir,
            self.log_dir,
        ])

        if self.config:
            self.load_yaml(self.config)

        
    def load_yaml(self, yaml_file: str) -> None:
        """Load config values from a YAML file under CONFIGS_PATH."""
        config_path = os.path.join(self.CONFIGS_PATH, yaml_file)

        with open(config_path, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file) or {}

        for key, value in yaml_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Recompute paths that depend on exp_name or other YAML-overridable fields
        self.log_dir = os.path.join(self.LOGS_PATH, f"tensorboard_{self.exp_name}")
        self.final_model_path = os.path.join(self.MODELS_PATH, f"tetris_turbomino_{self.exp_name}.zip")
        make_dirs([self.log_dir])

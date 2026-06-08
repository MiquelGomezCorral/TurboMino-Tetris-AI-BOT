"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
from dataclasses import dataclass, field

from maikol_utils.file_utils import make_dirs
from maikol_utils.print_utils import print_separator
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
    processed_dataset_path: str = os.path.join(DATA_PATH, "processed_dataset.csv")

    pretrain_model_path: str = os.path.join(MODELS_PATH, "pretrain_model")
    checkpoint_dir: str = os.path.join(MODELS_PATH, "checkpoints")
    log_dir: str = os.path.join(LOGS_PATH, f"tensorboard_{exp_name}")
    final_model_path: str = os.path.join(MODELS_PATH, f"tetris_turbomino_{exp_name}.zip")

    model_path: str = None
    # ===================================================================
    #                       PARAMETER PRETRAIN
    # ===================================================================

    seed:     int = 42
    test_size: float = 0.2
    val_size:  float = 0.1


    epochs: int = 100
    patience: int = 10
    label_smoothing: float = 0.1


    # ===================================================================
    #                       PARAMETER RL
    # ===================================================================

    gym_id:          str = None
    
    total_timesteps: int = 25_000
    max_placements: int = 128
    d_model: int = 64
    n_heads: int = 4
    head_hidden: int = 128
    n_piece_layers: int = 2
    max_board_size_w: int = 10
    max_board_size_h: int = 20



    net_arch: list[int] = field(default_factory=lambda: [64, 64])
    features_per_placement: int = 4
    learning_rate: float = 3e-4
    lr_end: float = 1e-5
    n_steps: int = 2048
    batch_size: int = 256
    ent_coef: float = 0.02
    ent_coef_end: float = 0.001
    clip_range: float = 0.2
    gamma: float = 0.999
    verbose: int = 0
    n_envs: int = 1

    save_freq: int = 50_000
    eval_episodes: int = 100
    max_eval_pieces: int = 100

    total_timesteps: int = 5_000_000

    clear_lines_on_placement: bool = True
    use_heuristic_rewards: bool = True


    def __post_init__(self):
        if self.config:
            self.load_yaml(self.config)

        # Recompute paths (exp_name may have changed from YAML or CLI)
        self.log_dir = os.path.join(self.LOGS_PATH, f"tensorboard_{self.exp_name}")
        self.final_model_path = os.path.join(self.MODELS_PATH, f"tetris_turbomino_{self.exp_name}.zip")
        if self.model_path is None:
            self.model_path = self.final_model_path

        make_dirs([
            self.DATA_PATH, 
            self.MODELS_PATH, 
            self.LOGS_PATH,
            self.checkpoint_dir,
            self.log_dir,
        ])

        
    def load_yaml(self, yaml_file: str) -> None:
        """Load config values from a YAML file under CONFIGS_PATH."""
        config_path = os.path.join(self.CONFIGS_PATH, yaml_file)

        with open(config_path, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file) or {}

        for key, value in yaml_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def print_config(self):
        print_separator("NORMAL CONFIG", sep_type="SHORT")

        for field_name, value in self.__dict__.items():
            print(f"- {field_name}: {value}")
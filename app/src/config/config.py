"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
from dataclasses import dataclass

from maikol_utils.file_utils import make_dirs
import yaml

@dataclass 
class Configuration:
    """Configuration class for the project."""
    # ===================================================================
    #                       PATHS
    # ===================================================================
    DATA_PATH: str = os.path.join("..", "data")
    MODELS_PATH: str = os.path.join("..", "models")
    LOGS_PATH: str = os.path.join("..", "logs")
    yaml_config_path: str = None

    raw_dataset_path: str = os.path.join(DATA_PATH, "data.csv")

    # ===================================================================
    #                       PARAMETER
    # ===================================================================

    exp_name: str = "base_name"
    seed:     int = 42

    max_placements: int = 50
    board_w: int = 10
    board_h: int = 20


    gym_id:          str = None
    total_timesteps: int = 25_000

    def __post_init__(self):
        # Basic setup: create folders and load yaml config if provided
        make_dirs([self.DATA_PATH, self.MODELS_PATH, self.LOGS_PATH])
        if self.yaml_config_path:
            self._load_yaml_configuration(self.yaml_config_path)

        # More stuff 
        ...

        
    def _load_yaml_configuration(self, yaml_file: str) -> None:
        """Load config values from a YAML file under MODELS_PATH."""
        config_path = os.path.join(self.MODELS_PATH, yaml_file)

        with open(config_path, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file) or {}

        for key, value in yaml_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

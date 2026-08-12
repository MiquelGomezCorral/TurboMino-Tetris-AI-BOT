"""Configuration file.

Configuration of project variables that we want to have available
everywhere and considered configuration.
"""
import os
import math
from dataclasses import dataclass, field

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
    tetrio_train: str = os.path.join(DATA_PATH, "tetrio_train.csv")
    tetrio_test: str = os.path.join(DATA_PATH, "tetrio_test.csv")
    tetrio_val: str = os.path.join(DATA_PATH, "tetrio_val.csv")
    precomputed_data: str = os.path.join(DATA_PATH, "precomputed")
    precomputed_train: str = os.path.join(precomputed_data, "train")
    precomputed_val: str = os.path.join(precomputed_data, "val")
    precomputed_test: str = os.path.join(precomputed_data, "test")

    pretrain_model_path: str = os.path.join(MODELS_PATH, "pretrain_model")
    checkpoint_dir: str = os.path.join(MODELS_PATH, "checkpoints", exp_name)
    log_dir: str = os.path.join(LOGS_PATH, f"tensorboard_{exp_name}")
    final_model_path: str = os.path.join(MODELS_PATH, f"tetris_turbomino_{exp_name}.zip")
    best_model_path: str = os.path.join(MODELS_PATH, "checkpoints", exp_name, "best_model.zip")

    model_path: str = None
    resume_model_path: str = None
    # ===================================================================
    #                       PARAMETER PRETRAIN
    # ===================================================================

    seed:     int = 42
    test_size: float = 0.10
    val_size:  float = 0.05
    batch_size: int = 256
    n_epochs: int = 10
    num_workers: int = 4

    tetrio_epochs: int = 100
    patience: int = 10
    label_smoothing: float = 0.1

    weight_decay: float = 1e-4
    eta_min: float = 1e-5

    aug_prob: float = 0.5

    wide_k: int = 4
    channels: int = 32

    # ===================================================================
    #                       PARAMETER RL
    # ===================================================================

    gym_id:          str = None
    
    max_placements: int = 156
    d_model: int = 156
    n_heads: int = 4
    head_hidden: int = 156
    n_piece_layers: int = 2
    max_board_size_w: int = 10
    max_board_size_h: int = 20



    net_arch: list[int] = field(default_factory=lambda: [156])
    features_per_placement: int = 4
    learning_rate: float = 3e-4
    rollout_samples: int = 2_048
    ent_coef: float = 0.02
    clip_range: float = 0.2
    gamma: float = 0.999
    gae_lambda: float = 0.98
    verbose: int = 0
    n_envs: int = 1
    target_kl: float = 0.02

    eval_every_rollouts: int = 1
    eval_episodes: int = 100
    eval_seed: int = 10_000
    max_eval_pieces: int = 200
    run_final_eval: bool = False
    curriculum_learned_ratio: float = 0.9
    curriculum_min_eval_score: float = 1_000.0

    total_timesteps: int = 5_000_000

    clear_lines_on_placement: bool = True # On the previsualization we send to the model!
    use_survival_rewards: bool = True
    use_heuristic_rewards: bool = False
    use_game_rewards: bool = False
    alive_reward: float = 0.1
    death_penalty: float = -5.0
    heuristic_reward_scale: float = 0.01
    heuristic_reward_cap: float = 0.1
    line_clear_reward: float = 0.1
    all_clear_reward: float = 0.4
    t_spin_reward: float = 0.2

    garbage_prob: float = 0.13
    garbage_delay: int = 5
    garbage_cap: int = 8
    garbage_lines_probs: list[float] = field(default_factory=lambda: [
        0.244018,
        0.147745,
        0.147703,
        0.146281,
        0.098800,
        0.096634,
        0.081859,
        0.036961,
    ])
    
    curriculum: dict = field(default_factory=dict)  # {board_w: timesteps}, e.g. {4: 1_000_000, 6: 1_000_000, 8: 1_000_000, 10: 2_000_000}
    random_width: dict = None

    def rollout_steps(self) -> int:
        return max(1, math.ceil(self.rollout_samples / self.n_envs))

    def eval_steps(self) -> int:
        return self.rollout_steps() * self.eval_every_rollouts


    def __post_init__(self):
        if self.config:
            self.load_yaml(self.config)

        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be at least 1")
        if self.n_epochs < 1:
            raise ValueError("n_epochs must be at least 1")
        if self.tetrio_epochs < 1:
            raise ValueError("tetrio_epochs must be at least 1")
        if self.eval_every_rollouts < 1:
            raise ValueError("eval_every_rollouts must be at least 1")
        if self.random_width:
            if any(not isinstance(width, int) or width < 1 or width > self.max_board_size_w for width in self.random_width):
                raise ValueError("random_width values must fit within max_board_size_w")
            if any(probability < 0 for probability in self.random_width.values()) or not math.isclose(sum(self.random_width.values()), 1.0):
                raise ValueError("random_width probabilities must sum to 1")

        # Recompute paths (exp_name may have changed from YAML or CLI)
        self.log_dir = os.path.join(self.LOGS_PATH, f"tensorboard_{self.exp_name}")
        self.checkpoint_dir = os.path.join(self.MODELS_PATH, "checkpoints", self.exp_name)
        self.final_model_path = os.path.join(self.MODELS_PATH, f"tetris_turbomino_{self.exp_name}.zip")
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.zip")
        if self.model_path is None:
            self.model_path = self.final_model_path

        
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

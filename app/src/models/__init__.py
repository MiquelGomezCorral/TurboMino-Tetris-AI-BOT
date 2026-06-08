"""Models.

Functions to manage, create, train / test models.
"""
from .gym_env import TetrisEnv
from .TurboMino import TurboMinoEncoder, TurboMinoModule
from .train_ppo import train_ppo_turbomino
from .train_tetrio import train_tetrio_turbomino, test_tetrio_turbomino
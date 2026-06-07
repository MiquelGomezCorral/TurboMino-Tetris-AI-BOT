"""Main file for scripts with arguments and call other functions."""

import dotenv
import argparse
from src.config import Configuration
from src.tetris import TetrisConfiguration
from maikol_utils.other_utils import args_to_dataclass
from maikol_utils.print_utils import print_separator

from scripts import play_tetris_game, showcase_model
from src.models import train_ppo_turbomino

def cmd_play_tetris(args: argparse.Namespace):
    """Call play_tetris_from_config_list with the given args."""
    CONFIG: TetrisConfiguration = args_to_dataclass(args, TetrisConfiguration)
    print_separator("START ...", sep_type="START")
    play_tetris_game(CONFIG)
    print_separator("END ...", sep_type="START")

def cmd_train(args):
    """Call training functions."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    T_CONFIG: TetrisConfiguration = args_to_dataclass(args, TetrisConfiguration)
    train_ppo_turbomino(CONFIG, T_CONFIG)

def cmd_showcase(args):
    """Call showcase functions."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    T_CONFIG: TetrisConfiguration = args_to_dataclass(args, TetrisConfiguration)
    showcase_model(CONFIG, T_CONFIG)

# ======================================================================================
#                                       ARGUMENTS
# ======================================================================================
if __name__ == "__main__":
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(prog="app", description="Main Application CLI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--config", type=str, default=None, help="Name of the config file at configs/ (default: None, but config.yaml exists)")
    subparsers = parser.add_subparsers(dest="function", required=True)

    # ======================================================================================
    #                                       play_tetris
    # ======================================================================================
    p_play = subparsers.add_parser("play-tetris", help="Play Tetris")
    p_play.add_argument("-W","--board_w", type=int, default=10, help="Board width (default: 10)")
    p_play.add_argument("-H","--board_h", type=int, default=20, help="Board height (default: 20)")
    p_play.set_defaults(func=cmd_play_tetris)

    # ======================================================================================
    #                                       train
    # ======================================================================================
    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.set_defaults(func=cmd_train)


     # ======================================================================================
    #                                       showcase
    # ======================================================================================
    p_showcase = subparsers.add_parser("showcase", help="Showcase the trained model")
    p_showcase.add_argument("--exp_name", type=str, default="base_name", help="Experiment name for logging and model saving (default: base_name)")
    p_showcase.add_argument("--model_path", type=str, default=None, help="Path to the trained model (default: None)")
    
    p_showcase.set_defaults(func=cmd_showcase)
    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)

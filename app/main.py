"""Main file for scripts with arguments and call other functions."""

import dotenv
import argparse
from maikol_utils.other_utils import args_to_dataclass
from maikol_utils.print_utils import print_separator

from src.config import Configuration
from src.tetris import TetrisConfiguration
from scripts import showcase_model, play_tetris_game
from src.models import train_ppo_turbomino, train_tetrio_turbomino

def cmd_play_tetris(args: argparse.Namespace):
    """Call play_tetris_from_config_list with the given args."""

    T_CONFIG: TetrisConfiguration = args_to_dataclass(args, TetrisConfiguration)
    print_separator("START TETRIS", sep_type="START")

    play_tetris_game(CONFIG=T_CONFIG)
    print_separator("END TETRIS", sep_type="START")

def cmd_train_ppo(args):
    """Call training functions."""
    print_separator("START TRAINING PPO", sep_type="START")

    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    T_CONFIG: TetrisConfiguration = args_to_dataclass(args, TetrisConfiguration)
    train_ppo_turbomino(CONFIG, T_CONFIG)
    print_separator("END TRAINING PPO", sep_type="START")

def cmd_train_tetrio(args):
    """Call training functions."""
    print_separator("START TRAINING TETRIO", sep_type="START")
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    T_CONFIG: TetrisConfiguration = args_to_dataclass(args, TetrisConfiguration)
    train_tetrio_turbomino(CONFIG, T_CONFIG)
    print_separator("END TRAINING TETRIO", sep_type="START")

def cmd_showcase(args):
    """Call showcase functions."""
    print_separator("START SHOWCASE", sep_type="START")
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    T_CONFIG: TetrisConfiguration = args_to_dataclass(args, TetrisConfiguration)
    showcase_model(CONFIG, T_CONFIG, use_ui=getattr(args, "ui", False))

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
    p_play.add_argument("--agent", action="store_true", help="Run the trained agent in the UI instead of human controls")
    p_play.add_argument("--exp_name", type=str, default=None, help="Experiment name for the agent model (default: None)")
    p_play.add_argument("--model_path", type=str, default=None, help="Path to the trained agent model (default: None)")
    p_play.set_defaults(func=cmd_play_tetris)

    # ======================================================================================
    #                                       train_ppo
    # ======================================================================================
    p_train = subparsers.add_parser("train-ppo", help="Train the PPO model")
    p_train.add_argument("-W","--board_w", type=int, default=10, help="Board width (default: 10)")
    p_train.add_argument("-H","--board_h", type=int, default=20, help="Board height (default: 20)")
    p_train.add_argument("-MP", "--resume_model_path", type=str, default=None, help="PPO checkpoint to resume (default: fresh training)")
    p_train.add_argument("--garbage_prob", type=float, default=None, help=f"Chance of receiving garbage after each placement (default: {Configuration.garbage_prob})")
    p_train.set_defaults(func=cmd_train_ppo)

    # ======================================================================================
    #                                       train_tetrio
    # ======================================================================================
    p_train = subparsers.add_parser("train-tetrio", help="Train the Tetrio model")
    p_train.set_defaults(func=cmd_train_tetrio)


    # ======================================================================================
    #                                       showcase
    # ======================================================================================
    p_showcase = subparsers.add_parser("showcase", help="Showcase the trained model")
    p_showcase.add_argument("--exp_name", type=str, default="base_name", help="Experiment name for logging and model saving (default: base_name)")
    p_showcase.add_argument("--model_path", type=str, default=None, help="Path to the trained model (default: None)")
    p_showcase.add_argument("--ui", action="store_true", help="Render the agent in the Pygame window instead of the terminal")
    p_showcase.add_argument("-W","--board_w", type=int, default=10, help="Board width (default: 10)")
    p_showcase.add_argument("-H","--board_h", type=int, default=20, help="Board height (default: 20)")
    p_showcase.add_argument("--garbage_prob", type=float, default=None, help=f"Chance of receiving garbage after each placement (default: {Configuration.garbage_prob})")
    p_showcase.set_defaults(func=cmd_showcase)
    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)

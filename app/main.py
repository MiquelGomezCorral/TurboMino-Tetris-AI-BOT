"""Main file for scripts with arguments and call other functions."""

import dotenv
import argparse
from src.config import Configuration
from maikol_utils.other_utils import args_to_dataclass
from maikol_utils.print_utils import print_separator

from scripts import play_tetris_game

def cmd_play_tetris(args: argparse.Namespace):
    """Call play_tetris_from_config_list with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    print_separator("START ...", sep_type="START")
    play_tetris_game()
    print_separator("END ...", sep_type="START")

def cmd_test(args):
    """Call test functions."""
    ...

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
    p_play.set_defaults(func=cmd_play_tetris)

    # ======================================================================================
    #                                       test
    # ======================================================================================
    p_test = subparsers.add_parser("test", help="Test script with any code")
    p_test.set_defaults(func=cmd_test)

    # ======================================================================================
    #                                       CALL
    # ======================================================================================
    args = parser.parse_args()
    args.func(args)

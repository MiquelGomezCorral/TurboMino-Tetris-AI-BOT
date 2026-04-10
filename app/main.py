"""Main file for scripts with arguments and call other functions."""

import dotenv
import argparse
from src.config import Configuration
from maikol_utils.other_utils import args_to_dataclass
from maikol_utils.print_utils import print_separator

def cmd_read_extract(args: argparse.Namespace):
    """Call read_extract_from_config_list with the given args."""
    CONFIG: Configuration = args_to_dataclass(args, Configuration)
    print_separator("START ...", sep_type="START")
    ...
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
    #                                       read_extract
    # ======================================================================================
    p_read = subparsers.add_parser("read-extract", help="Read and extract from config list")
    p_read.add_argument(
        "-d", "--dataset_name", type=str, default="Nuelas", help="Name of raw data folder"
    )
    p_read.add_argument("-m", "--max_files", type=int, default=None, help="Max files to load")
    p_read.add_argument(
        "-l", "--use_llm", action="store_false", default=True, help="Disable LLM extraction"
    )
    p_read.set_defaults(func=cmd_read_extract)

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

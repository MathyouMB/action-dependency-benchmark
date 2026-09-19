import argparse
from pathlib import Path

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to a configuration JSON file")
    args = parser.parse_args()
    if not args.config:
        print(f"{RED}Error: --config is required{RESET}")
        raise SystemExit(2)
    return args


def validate_config(path: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"config file not found: {path}")


def main() -> None:
    """Step 1. validate all arguments are present."""
    args = parse_args()

    """Step 2. validate all the argument values are valid."""
    try:
        validate_config(args.config)
    except FileNotFoundError as exception:
        print(f"{RED}Error: {exception}{RESET}")
        raise SystemExit(1)
    print(f"{GREEN}Hello, world! config={args.config}{RESET}")


if __name__ == "__main__":
    main()

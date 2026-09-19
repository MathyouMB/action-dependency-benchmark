import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a configuration JSON file")
    args = parser.parse_args()
    validate_config(args.config)
    print(f"Hello, world! config={args.config}")


def validate_config(path: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"config file not found: {path}")


if __name__ == "__main__":
    main()

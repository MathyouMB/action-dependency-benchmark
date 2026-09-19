import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.models.runners.base import Instance, Runner
from benchmark.models.runners.gpt_oss_ollama import GptOssOllamaRunner

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_TASK = REPO_ROOT / "data" / "tasks" / "webarena_000284" / "webarena_000284_v0001.json"

RUNNERS: dict[str, type[Runner]] = {
    "gpt_oss_ollama": GptOssOllamaRunner,
}


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


def build_runner(config: dict[str, Any]) -> Runner:
    runner_cls = RUNNERS.get(config["runner"])
    if runner_cls is None:
        raise ValueError(f"unknown runner {config['runner']!r}; available: {sorted(RUNNERS)}")
    return runner_cls(config)


def main() -> None:
    """Step 1. validate all arguments are present."""
    args = parse_args()

    """Step 2. validate all the argument values are valid."""
    try:
        validate_config(args.config)
    except FileNotFoundError as exception:
        print(f"{RED}Error: {exception}{RESET}")
        raise SystemExit(1) from None

    """Step 3. build the configured runner and evoke it."""
    try:
        config = json.loads(Path(args.config).read_text())
        runner = build_runner(config)
        instance = Instance.from_file(SAMPLE_TASK)
        response = runner.predict(instance)
    except Exception as exception:
        print(f"{RED}Error: {type(exception).__name__}: {exception}{RESET}")
        raise SystemExit(1) from None

    print(f"{GREEN}{response}{RESET}")


if __name__ == "__main__":
    main()

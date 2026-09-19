import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.models.runners.base import Instance, Runner, TaskResult
from benchmark.models.runners.gpt_oss_ollama import GptOssOllamaRunner

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS_DIR = REPO_ROOT / "data" / "tasks"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

RUNNERS: dict[str, type[Runner]] = {
    "gpt_oss_ollama": GptOssOllamaRunner,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to a configuration JSON file")
    parser.add_argument(
        "--tasks",
        default=DEFAULT_TASKS_DIR,
        help="a task file, a scenario folder, or a directory searched recursively "
        "for task files (default: %(default)s)",
    )
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


def load_instances(tasks_path: str | Path) -> list[Instance]:
    """Every task file under `tasks_path`, in a stable order.

    `tasks_path` may be a single task file, a scenario folder holding a few, or
    the whole tasks directory - it is searched recursively either way.
    """
    tasks_path = Path(tasks_path)
    if tasks_path.is_file():
        return [Instance.from_file(tasks_path)]
    return [Instance.from_file(path) for path in sorted(tasks_path.rglob("*.json"))]


def default_output_path(config_name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_RESULTS_DIR / config_name / f"{stamp}.jsonl"


def predict_safely(runner: Runner, instance: Instance) -> TaskResult:
    try:
        return runner.predict(instance)
    except Exception as exception:
        return TaskResult(
            predicted_ids=None,
            expected_ids=instance.supporting_fact_ids,
            error=f"{type(exception).__name__}: {exception}",
        )


def main() -> None:
    """Step 1. validate all arguments are present."""
    args = parse_args()

    """Step 2. validate all the argument values are valid."""
    try:
        validate_config(args.config)
    except FileNotFoundError as exception:
        print(f"{RED}Error: {exception}{RESET}")
        raise SystemExit(1) from None

    """Step 3. build the configured runner and load its scenarios."""
    try:
        config = json.loads(Path(args.config).read_text())
        config.setdefault("name", Path(args.config).stem)
        runner = build_runner(config)
        instances = load_instances(args.tasks)
    except Exception as exception:
        print(f"{RED}Error: {type(exception).__name__}: {exception}{RESET}")
        raise SystemExit(1) from None

    """Step 4. evoke the runner on every scenario, recording each as it finishes."""
    output_path = default_output_path(config["name"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"{config['name']} ({config['runner']}) over {len(instances)} instance(s)")
    with output_path.open("w") as handle:
        for instance in instances:
            result = predict_safely(runner, instance)
            record = {
                "instance_id": instance.instance_id,
                "config_name": config["name"],
                "runner": config["runner"],
                "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                **asdict(result),
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()

            color = RED if result.error else GREEN
            outcome = result.error or result.predicted_ids
            print(f"  {color}{instance.instance_id}  {outcome}{RESET}")

    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

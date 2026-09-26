import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.env import load_env_file
from benchmark.models.runners.base import Instance, Runner, TaskResult
from benchmark.models.runners.deepseek_r1_ollama import DeepSeekR1OllamaRunner
from benchmark.models.runners.gpt_oss_ollama import GptOssOllamaRunner
from benchmark.models.runners.ollama_chat import OllamaChatRunner
from benchmark.models.runners.openrouter_chat import OpenRouterChatRunner
from benchmark.models.runners.qwen3_ollama import Qwen3OllamaRunner
from benchmark.models.view import RED, RESET, RunView

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TASKS_DIR = REPO_ROOT / "data" / "tasks"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"

RUNNERS: dict[str, type[Runner]] = {
    "gpt_oss_ollama": GptOssOllamaRunner,
    "deepseek_r1_ollama": DeepSeekR1OllamaRunner,
    "qwen3_ollama": Qwen3OllamaRunner,
    "ollama": OllamaChatRunner,
    "openrouter": OpenRouterChatRunner,
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


def run_output_path(config_name: str, stamp: str) -> Path:
    """Every instance of one run, a record per line."""
    return DEFAULT_RESULTS_DIR / "runs" / config_name / f"{stamp}.jsonl"


def task_output_path(instance: Instance, config_name: str, stamp: str) -> Path:
    """One instance on its own, filed under the task it is a version of."""
    return (
        DEFAULT_RESULTS_DIR
        / "tasks"
        / instance.scenario_id
        / instance.instance_id
        / config_name
        / f"{stamp}.json"
    )


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
    load_env_file()
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
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = run_output_path(config["name"], stamp)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    view = RunView(config["name"], config["runner"], len(instances))
    run_started = time.monotonic()
    with output_path.open("w") as handle:
        for index, instance in enumerate(instances, start=1):
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

            task_path = task_output_path(instance, config["name"], stamp)
            task_path.parent.mkdir(parents=True, exist_ok=True)
            task_path.write_text(json.dumps(record, indent=2) + "\n")

            view.print_instance(index, instance, result)
    view.print_summary(time.monotonic() - run_started)

    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

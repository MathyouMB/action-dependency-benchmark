import json
import sys

import pytest

from benchmark.models import runner as runner_module
from benchmark.models.runner import (
    load_instances,
    main,
    parse_args,
    validate_config,
)
from benchmark.models.runners.base import Instance, Runner, TaskResult
from benchmark.models.runners.openrouter_chat import OpenRouterChatRunner
from benchmark.models.view import GREEN, RED


class FakeRunner(Runner):
    def predict(self, instance: Instance) -> TaskResult:
        return TaskResult(predicted_ids=["f1"], expected_ids=instance.supporting_fact_ids)


class ExplodingRunner(Runner):
    def predict(self, instance: Instance) -> TaskResult:
        raise RuntimeError("connection refused")


def write_task(root, instance_id, supporting=("f1",)):
    directory = root / instance_id.rsplit("_v", 1)[0]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{instance_id}.json"
    path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "scenario_id": instance_id.rsplit("_v", 1)[0],
                "proposed_action": {"text": "Open the product page"},
                "state": [{"id": "f1", "text": "a"}, {"id": "f2", "text": "b"}],
                "supporting_fact_ids": list(supporting),
                "webarena_data": {"intent": "an intent"},
            }
        )
    )
    return path


def test_validate_config_raises_if_the_file_does_not_exist(tmp_path):
    missing = tmp_path / "nope.json"

    with pytest.raises(FileNotFoundError, match="nope.json"):
        validate_config(missing)


def test_validate_config_passes_for_an_existing_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{}")

    validate_config(path)


def test_parse_args_returns_the_given_config_path(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runner.py", "--config", "foo.json"])

    args = parse_args()

    assert args.config == "foo.json"


def test_parse_args_exits_with_a_red_error_when_config_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["runner.py"])

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
    out = capsys.readouterr().out
    assert RED in out
    assert "--config is required" in out


def test_load_instances_accepts_a_single_task_file(tmp_path):
    path = write_task(tmp_path, "webarena_000001_v0001")

    instances = load_instances(path)

    assert [i.instance_id for i in instances] == ["webarena_000001_v0001"]


def test_load_instances_finds_every_task_file_under_a_directory(tmp_path):
    write_task(tmp_path, "webarena_000001_v0001")
    write_task(tmp_path, "webarena_000002_v0001")

    instances = load_instances(tmp_path)

    assert [i.instance_id for i in instances] == [
        "webarena_000001_v0001",
        "webarena_000002_v0001",
    ]


def test_main_writes_one_result_line_per_instance(monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(runner_module.RUNNERS, "fake", FakeRunner)
    monkeypatch.setattr(runner_module, "DEFAULT_RESULTS_DIR", tmp_path / "results")
    tasks = tmp_path / "tasks"
    write_task(tasks, "webarena_000001_v0001", supporting=["f1"])
    write_task(tasks, "webarena_000002_v0001", supporting=["f2"])
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"runner": "fake"}))
    monkeypatch.setattr(sys, "argv", ["runner.py", "--config", str(config), "--tasks", str(tasks)])

    main()

    out = capsys.readouterr().out
    assert GREEN in out
    assert "webarena_000001_v0001" in out
    assert "webarena_000002_v0001" in out

    written = list((tmp_path / "results" / "runs" / "config").glob("*.jsonl"))
    assert len(written) == 1
    records = [json.loads(line) for line in written[0].read_text().splitlines()]
    assert [r["instance_id"] for r in records] == [
        "webarena_000001_v0001",
        "webarena_000002_v0001",
    ]
    assert all(r["predicted_ids"] == ["f1"] for r in records)


def test_main_writes_each_result_under_its_task_and_version(monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(runner_module.RUNNERS, "fake", FakeRunner)
    monkeypatch.setattr(runner_module, "DEFAULT_RESULTS_DIR", tmp_path / "results")
    tasks = tmp_path / "tasks"
    write_task(tasks, "webarena_000001_v0001", supporting=["f1"])
    write_task(tasks, "webarena_000001_v0002", supporting=["f2"])
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"runner": "fake"}))
    monkeypatch.setattr(sys, "argv", ["runner.py", "--config", str(config), "--tasks", str(tasks)])

    main()

    run_file = next((tmp_path / "results" / "runs" / "config").glob("*.jsonl"))
    lines = [json.loads(line) for line in run_file.read_text().splitlines()]
    scenario = tmp_path / "results" / "tasks" / "webarena_000001"

    expected_ids = ["webarena_000001_v0001", "webarena_000001_v0002"]
    for instance_id, line in zip(expected_ids, lines, strict=True):
        # the run's stamp names the per-task file too, so the two layouts line up
        path = scenario / instance_id / "config" / f"{run_file.stem}.json"
        assert json.loads(path.read_text()) == line
        assert line["instance_id"] == instance_id


def test_main_records_a_runner_error_without_losing_the_rest_of_the_run(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setitem(runner_module.RUNNERS, "exploding", ExplodingRunner)
    monkeypatch.setattr(runner_module, "DEFAULT_RESULTS_DIR", tmp_path / "results")
    tasks = tmp_path / "tasks"
    write_task(tasks, "webarena_000001_v0001")
    write_task(tasks, "webarena_000002_v0001")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"runner": "exploding"}))
    monkeypatch.setattr(sys, "argv", ["runner.py", "--config", str(config), "--tasks", str(tasks)])

    main()

    out = capsys.readouterr().out
    assert RED in out

    written = list((tmp_path / "results" / "runs" / "config").glob("*.jsonl"))
    records = [json.loads(line) for line in written[0].read_text().splitlines()]
    assert len(records) == 2
    assert all(r["predicted_ids"] is None for r in records)
    assert all("connection refused" in r["error"] for r in records)


def test_main_exits_with_a_red_error_for_a_missing_config_file(monkeypatch, capsys, tmp_path):
    missing = tmp_path / "nope.json"
    monkeypatch.setattr(sys, "argv", ["runner.py", "--config", str(missing)])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert RED in out
    assert "config file not found" in out


def test_build_runner_knows_the_openrouter_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("{{task}} {{state}} {{proposed_action}}")

    built = runner_module.build_runner(
        {
            "name": "x",
            "runner": "openrouter",
            "prompt": str(prompt),
            "args": {"model": "openai/gpt-oss-120b"},
        }
    )

    assert isinstance(built, OpenRouterChatRunner)


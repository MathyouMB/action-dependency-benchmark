import json
import sys

import pytest

from benchmark.models import runner as runner_module
from benchmark.models.runner import GREEN, RED, main, parse_args, validate_config
from benchmark.models.runners.base import Instance, Runner


class FakeRunner(Runner):
    def predict(self, instance: Instance) -> str:
        return f"predicted for {instance.instance_id}"


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


def test_main_prints_a_green_success_message(monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(runner_module.RUNNERS, "fake", FakeRunner)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"runner": "fake"}))
    monkeypatch.setattr(sys, "argv", ["runner.py", "--config", str(config)])

    main()

    out = capsys.readouterr().out
    assert GREEN in out
    assert (
        f"predicted for {runner_module.Instance.from_file(runner_module.SAMPLE_TASK).instance_id}"
        in out
    )


def test_main_exits_with_a_red_error_for_a_missing_config_file(monkeypatch, capsys, tmp_path):
    missing = tmp_path / "nope.json"
    monkeypatch.setattr(sys, "argv", ["runner.py", "--config", str(missing)])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert RED in out
    assert "config file not found" in out

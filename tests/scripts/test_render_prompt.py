import json
import sys

import pytest

from benchmark.models.view import RED
from scripts.render_prompt import main, render, resolve_out_path

TEMPLATE = """TASK THE USER ASKED FOR:
{{task}}

OBSERVED STATE:
{{state}}

PROPOSED ACTION:
{{proposed_action}}
"""


def write_instance(root, instance_id="webarena_000001_v0001"):
    path = root / f"{instance_id}.json"
    path.write_text(
        json.dumps(
            {
                "instance_id": instance_id,
                "scenario_id": instance_id.rsplit("_v", 1)[0],
                "proposed_action": {"text": "Book Hotel Cedar"},
                "state": [
                    {"id": "f1", "text": "Hotel Alba has free wifi"},
                    {"id": "f2", "text": "Hotel Cedar is rated 4.6"},
                ],
                "supporting_fact_ids": ["f1", "f2"],
                "webarena_data": {"intent": "Book the highest-rated hotel with free wifi"},
            }
        )
    )
    return path


def write_template(root, text=TEMPLATE):
    path = root / "demo.md"
    path.write_text(text)
    return path


def test_render_fills_the_task_state_and_action(tmp_path):
    instance = write_instance(tmp_path)
    template = write_template(tmp_path)

    rendered = render(instance, template)

    assert "Book the highest-rated hotel with free wifi" in rendered
    assert "f1: Hotel Alba has free wifi" in rendered
    assert "f2: Hotel Cedar is rated 4.6" in rendered
    assert "Book Hotel Cedar" in rendered


def test_render_does_not_leak_the_supporting_fact_ids(tmp_path):
    instance = write_instance(tmp_path)
    template = write_template(tmp_path)

    rendered = render(instance, template)

    assert "supporting_fact_ids" not in rendered
    # the ids appear only as state lines, never as an answer key
    assert rendered.count("f1") == 1
    assert rendered.count("f2") == 1


def test_render_raises_for_a_placeholder_the_instance_cannot_fill(tmp_path):
    instance = write_instance(tmp_path)
    template = write_template(tmp_path, "{{task}} and {{mystery}}")

    with pytest.raises(ValueError, match="mystery"):
        render(instance, template)


def test_resolve_out_path_defaults_to_the_rendered_dir_named_for_the_instance(tmp_path):
    instance = write_instance(tmp_path)

    path = resolve_out_path(None, instance)

    assert path.name == "webarena_000001_v0001.md"
    assert path.parent.name == "rendered"


def test_resolve_out_path_honours_an_explicit_out(tmp_path):
    instance = write_instance(tmp_path)

    path = resolve_out_path(str(tmp_path / "elsewhere.md"), instance)

    assert path == tmp_path / "elsewhere.md"


def test_main_writes_the_rendered_prompt_and_prints_where(monkeypatch, capsys, tmp_path):
    instance = write_instance(tmp_path)
    template = write_template(tmp_path)
    out = tmp_path / "out" / "prompt.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["render_prompt.py", str(instance), "--prompt", str(template), "--out", str(out)],
    )

    main()

    written = out.read_text()
    assert "f1: Hotel Alba has free wifi" in written
    assert "Book Hotel Cedar" in written
    assert str(out) in capsys.readouterr().out


def test_main_prints_the_prompt_itself_with_stdout(monkeypatch, capsys, tmp_path):
    instance = write_instance(tmp_path)
    template = write_template(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_prompt.py",
            str(instance),
            "--prompt",
            str(template),
            "--out",
            str(tmp_path / "prompt.md"),
            "--stdout",
        ],
    )

    main()

    assert "Book Hotel Cedar" in capsys.readouterr().out


def test_main_exits_with_a_red_error_for_a_missing_instance(monkeypatch, capsys, tmp_path):
    template = write_template(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["render_prompt.py", str(tmp_path / "nope.json"), "--prompt", str(template)],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert RED in out
    assert "nope.json" in out


def test_main_exits_with_a_red_error_for_a_missing_template(monkeypatch, capsys, tmp_path):
    instance = write_instance(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["render_prompt.py", str(instance), "--prompt", str(tmp_path / "nope.md")],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert RED in out
    assert "nope.md" in out

from dataclasses import asdict

import pytest

from benchmark.models.runners.base import TaskResult, render_json


def test_render_json_replaces_a_whole_value_with_a_typed_value():
    template = {"observed_state": "{{state_lines}}"}

    rendered = render_json(template, {"state_lines": ["f1: a", "f2: b"]})

    assert rendered == {"observed_state": ["f1: a", "f2: b"]}


def test_render_json_substitutes_a_placeholder_embedded_in_a_longer_string():
    template = {"instructions": "Is fact {{fact_id}} a dependency of the proposed action?"}

    rendered = render_json(template, {"fact_id": "f1"})

    assert rendered == {"instructions": "Is fact f1 a dependency of the proposed action?"}


def test_render_json_walks_nested_dicts_and_lists():
    template = {
        "state": {"task": "{{task}}", "exclusions": ["keep {{task}} out", "untouched"]},
        "question": {"criteria": {"true": "{{fact_id}} was read", "false": "not read"}},
    }

    rendered = render_json(template, {"task": "buy a thing", "fact_id": "f3"})

    assert rendered["state"]["task"] == "buy a thing"
    assert rendered["state"]["exclusions"] == ["keep buy a thing out", "untouched"]
    assert rendered["question"]["criteria"]["true"] == "f3 was read"
    assert rendered["question"]["criteria"]["false"] == "not read"


def test_render_json_leaves_non_string_leaves_alone():
    template = {"type": "noul", "threshold": 0.5, "enabled": True, "nothing": None}

    assert render_json(template, {}) == template


def test_render_json_raises_for_an_unfilled_placeholder():
    template = {"instructions": "Is fact {{fact_id}} a {{mystery}}?"}

    with pytest.raises(ValueError, match="mystery"):
        render_json(template, {"fact_id": "f1"})


def test_render_json_raises_for_an_unfilled_whole_value_placeholder():
    template = {"observed_state": "{{state_lines}}"}

    with pytest.raises(ValueError, match="state_lines"):
        render_json(template, {})


def test_render_json_does_not_mistake_braces_in_the_data_for_placeholders():
    """The check must read the template, not the filled output."""
    template = {"observed_state": "{{state_lines}}"}

    rendered = render_json(template, {"state_lines": ["f1: uses {{price}} in the cell"]})

    assert rendered == {"observed_state": ["f1: uses {{price}} in the cell"]}


def test_answers_defaults_to_none_so_existing_runners_are_unaffected():
    result = TaskResult(predicted_ids=["f1"], expected_ids=["f1"])

    assert result.answers is None


def test_answers_is_kept_verbatim_when_given():
    returned = {"f1_dependency": {"type": "noul", "noul": 0.95}}

    result = TaskResult(predicted_ids=["f1"], expected_ids=["f1"], answers=returned)

    assert result.answers == returned


def test_answers_reaches_the_written_record_via_asdict():
    result = TaskResult(
        predicted_ids=["f1"],
        expected_ids=["f1"],
        answers={"f1_dependency": {"type": "noul", "noul": 0.95}},
    )

    record = asdict(result)

    assert record["answers"] == {"f1_dependency": {"type": "noul", "noul": 0.95}}

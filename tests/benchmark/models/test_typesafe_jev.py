import io
import json
import urllib.error

import pytest

from benchmark.models.runners import typesafe_jev as jev_module
from benchmark.models.runners.base import Instance
from benchmark.models.runners.typesafe_jev import (
    ENDPOINT,
    TypeSafeJevRunner,
    build_request,
    fact_ids,
)

TEMPLATE = {
    "state": {
        "task": "{{task}}",
        "observed_state": "{{state_lines}}",
        "proposed_action": "{{proposed_action}}",
        "dependency_definition": "A dependency is a fact the reasoning had to read.",
        "exclusions": ["Do not include facts the task does not use."],
    },
    "question_key": "{{fact_id}}_dependency",
    "question": {
        "type": "noul",
        "instructions": "Is fact {{fact_id}} a dependency of the proposed action?",
        "criteria": {
            "true": "Fact {{fact_id}} had to be read.",
            "false": "Fact {{fact_id}} did not have to be read.",
        },
    },
}


def make_instance(state=None):
    return Instance(
        instance_id="webarena_000284_v0001",
        scenario_id="webarena_000284",
        task="buy the cheapest shoe rack that holds 12 pairs",
        state=state
        if state is not None
        else [
            {"id": "f1", "text": "MiniStack holds 9 pairs", "annotations": {"domain": "numeric"}},
            {"id": "f2", "text": "MiniStack costs $57.11", "annotations": {"domain": "numeric"}},
        ],
        proposed_action="Open the product page for ShoeNest",
        supporting_fact_ids=["f1"],
    )


def json_text(body):
    """The request body as one string, for asserting what is *not* in it."""
    return json.dumps(body)


def test_fact_ids_reads_them_in_state_order():
    assert fact_ids(make_instance()) == ["f1", "f2"]


def test_fact_ids_rejects_a_state_with_no_facts():
    with pytest.raises(ValueError, match="no observed facts"):
        fact_ids(make_instance(state=[]))


def test_fact_ids_rejects_duplicate_ids_rather_than_silently_asking_fewer_questions():
    state = [{"id": "f1", "text": "a"}, {"id": "f1", "text": "b"}, {"id": "f2", "text": "c"}]

    with pytest.raises(ValueError, match="duplicate fact id"):
        fact_ids(make_instance(state=state))


def test_it_renders_the_state_block_once_with_the_scenario():
    body, _ = build_request(TEMPLATE, make_instance())

    assert body["state"]["task"] == "buy the cheapest shoe rack that holds 12 pairs"
    assert body["state"]["proposed_action"] == "Open the product page for ShoeNest"
    assert body["state"]["observed_state"] == [
        "f1: MiniStack holds 9 pairs",
        "f2: MiniStack costs $57.11",
    ]


def test_it_keeps_the_decision_rule_and_exclusions_verbatim():
    body, _ = build_request(TEMPLATE, make_instance())

    definition = body["state"]["dependency_definition"]
    assert definition == "A dependency is a fact the reasoning had to read."
    assert body["state"]["exclusions"] == ["Do not include facts the task does not use."]


def test_it_never_sends_the_dataset_annotations():
    body, _ = build_request(TEMPLATE, make_instance())

    assert "annotations" not in json_text(body)
    assert "numeric" not in json_text(body)


def test_it_asks_exactly_one_question_per_fact():
    body, keys = build_request(TEMPLATE, make_instance())

    assert list(body["questions"]) == ["f1_dependency", "f2_dependency"]
    assert keys == {"f1_dependency": "f1", "f2_dependency": "f2"}


def test_every_question_is_a_noul_naming_its_own_fact():
    body, _ = build_request(TEMPLATE, make_instance())

    assert body["questions"]["f1_dependency"] == {
        "type": "noul",
        "instructions": "Is fact f1 a dependency of the proposed action?",
        "criteria": {
            "true": "Fact f1 had to be read.",
            "false": "Fact f1 did not have to be read.",
        },
    }


def test_question_wording_is_identical_across_facts_but_for_the_fact_id():
    body, _ = build_request(TEMPLATE, make_instance())

    first = json_text(body["questions"]["f1_dependency"]).replace("f1", "ID")
    second = json_text(body["questions"]["f2_dependency"]).replace("f2", "ID")
    assert first == second


def test_it_does_not_leak_the_supporting_fact_ids():
    body, _ = build_request(TEMPLATE, make_instance())

    assert "supporting_fact_ids" not in json_text(body)


def make_config(tmp_path, template=None, **args):
    prompt = tmp_path / "jev.json"
    prompt.write_text(json.dumps(template if template is not None else TEMPLATE))
    return {
        "name": "typesafe-jev-test",
        "runner": "typesafe_jev",
        "prompt": str(prompt),
        "args": {"model": "jev-1.13.0", **args},
    }


def response_body(answers=None, **overrides):
    body = {
        "model": "jev-1.13.0",
        "answers": answers
        if answers is not None
        else {
            "f1_dependency": {"type": "noul", "noul": 0.95},
            "f2_dependency": {"type": "noul", "noul": 0.05},
        },
        "usage": {"input_tokens": 1200, "output_tokens": 20},
    }
    body.update(overrides)
    return body


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class FakeUrlopen:
    """Stands in for urllib.request.urlopen, recording the request it was given."""

    def __init__(self, body):
        self.body = body
        self.request = None
        self.timeout = None

    def __call__(self, request, timeout=None):
        self.request = request
        self.timeout = timeout
        return FakeResponse(json.dumps(self.body).encode())


@pytest.fixture
def fake_urlopen(monkeypatch):
    def install(body):
        fake = FakeUrlopen(body)
        monkeypatch.setattr(jev_module.urllib.request, "urlopen", fake)
        return fake

    return install


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")


def sent_payload(fake):
    return json.loads(fake.request.data)


def test_it_refuses_to_build_without_an_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        TypeSafeJevRunner(make_config(tmp_path))


def test_it_posts_to_the_systemone_endpoint_with_a_bearer_token(tmp_path, fake_urlopen):
    fake = fake_urlopen(response_body())

    TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())

    assert fake.request.full_url == "https://api.typesafe.ai/v1/systemone"
    assert fake.request.headers["Authorization"] == "Bearer ts-test"


def test_it_sends_the_model_state_and_one_question_per_fact_in_one_request(tmp_path, fake_urlopen):
    fake = fake_urlopen(response_body())

    TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())

    payload = sent_payload(fake)
    assert payload["model"] == "jev-1.13.0"
    assert payload["state"]["task"] == "buy the cheapest shoe rack that holds 12 pairs"
    assert list(payload["questions"]) == ["f1_dependency", "f2_dependency"]


def test_it_thresholds_the_probabilities_into_a_prediction(tmp_path, fake_urlopen):
    fake_urlopen(response_body())

    result = TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())

    assert result.predicted_ids == ["f1"]
    assert result.expected_ids == ["f1"]


def test_a_probability_of_exactly_the_threshold_counts_as_a_dependency(tmp_path, fake_urlopen):
    fake_urlopen(
        response_body(
            answers={
                "f1_dependency": {"type": "noul", "noul": 0.5},
                "f2_dependency": {"type": "noul", "noul": 0.49},
            }
        )
    )

    result = TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())

    assert result.predicted_ids == ["f1"]


def test_it_honours_a_configured_threshold(tmp_path, fake_urlopen):
    fake_urlopen(response_body())

    result = TypeSafeJevRunner(make_config(tmp_path, threshold=0.99)).predict(make_instance())

    assert result.predicted_ids == []


def test_it_maps_usage_onto_the_token_counts(tmp_path, fake_urlopen):
    fake_urlopen(response_body())

    result = TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())

    assert result.prompt_eval_count == 1200
    assert result.eval_count == 20


def test_it_records_a_result_when_the_reply_omits_usage(tmp_path, fake_urlopen):
    body = response_body()
    del body["usage"]
    fake_urlopen(body)

    result = TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())

    assert result.predicted_ids == ["f1"]
    assert result.prompt_eval_count is None
    assert result.eval_count is None


def test_it_raises_when_a_fact_has_no_answer(tmp_path, fake_urlopen):
    fake_urlopen(response_body(answers={"f1_dependency": {"type": "noul", "noul": 0.95}}))

    with pytest.raises(RuntimeError, match="f2"):
        TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())


def test_it_raises_when_an_answer_carries_no_noul_value(tmp_path, fake_urlopen):
    fake_urlopen(
        response_body(
            answers={
                "f1_dependency": {"type": "noul", "noul": 0.95},
                "f2_dependency": {"type": "noul"},
            }
        )
    )

    with pytest.raises(RuntimeError, match="f2"):
        TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())


def test_it_ignores_answers_to_questions_it_never_asked(tmp_path, fake_urlopen):
    fake_urlopen(
        response_body(
            answers={
                "f1_dependency": {"type": "noul", "noul": 0.95},
                "f2_dependency": {"type": "noul", "noul": 0.05},
                "f9_dependency": {"type": "noul", "noul": 0.99},
            }
        )
    )

    result = TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())

    assert result.predicted_ids == ["f1"]


def test_it_records_the_version_jev_reports_and_never_asks_why(tmp_path, fake_urlopen):
    fake_urlopen(response_body())

    result = TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())

    assert result.meta["model"] == "jev-1.13.0"
    assert result.meta["threshold"] == 0.5
    assert "why" not in result.meta


def test_it_times_the_call_itself(tmp_path, fake_urlopen, monkeypatch):
    fake_urlopen(response_body())
    ticks = iter([100.0, 102.5])
    monkeypatch.setattr(jev_module.time, "monotonic", lambda: next(ticks))

    result = TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())

    assert result.total_duration_s == 2.5


def test_it_honours_the_configured_timeout(tmp_path, fake_urlopen):
    fake = fake_urlopen(response_body())

    TypeSafeJevRunner(make_config(tmp_path, timeout_s=42)).predict(make_instance())

    assert fake.timeout == 42


def test_it_refuses_a_question_key_that_does_not_vary_by_fact(tmp_path, fake_urlopen):
    """A template that dropped its {{fact_id}} would ask one question, not N."""
    fake_urlopen(response_body())
    template = json.loads(json.dumps(TEMPLATE))
    template["question_key"] = "dependency"

    with pytest.raises(ValueError, match="collapse"):
        TypeSafeJevRunner(make_config(tmp_path, template=template)).predict(make_instance())


def test_it_raises_when_a_noul_value_is_null(tmp_path, fake_urlopen):
    fake_urlopen(
        response_body(
            answers={
                "f1_dependency": {"type": "noul", "noul": 0.95},
                "f2_dependency": {"type": "noul", "noul": None},
            }
        )
    )

    with pytest.raises(RuntimeError, match="f2"):
        TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())


def test_it_raises_when_a_noul_value_is_not_a_number(tmp_path, fake_urlopen):
    fake_urlopen(
        response_body(
            answers={
                "f1_dependency": {"type": "noul", "noul": 0.95},
                "f2_dependency": {"type": "noul", "noul": "0.95"},
            }
        )
    )

    with pytest.raises(RuntimeError, match="f2"):
        TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())


def test_it_raises_when_a_noul_value_is_a_boolean(tmp_path, fake_urlopen):
    """True >= 0.5 is True in Python; a boolean must not read as certainty."""
    fake_urlopen(
        response_body(
            answers={
                "f1_dependency": {"type": "noul", "noul": 0.95},
                "f2_dependency": {"type": "noul", "noul": True},
            }
        )
    )

    with pytest.raises(RuntimeError, match="f2"):
        TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())


def test_it_raises_when_an_answer_is_not_an_object(tmp_path, fake_urlopen):
    fake_urlopen(
        response_body(
            answers={"f1_dependency": {"type": "noul", "noul": 0.95}, "f2_dependency": 0.05}
        )
    )

    with pytest.raises(RuntimeError, match="f2"):
        TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())


def test_it_surfaces_an_api_error_returned_in_a_200_body(tmp_path, fake_urlopen):
    """Otherwise a rate-limited run records N identical phantom parse errors."""
    fake_urlopen({"error": {"code": 429, "message": "rate limited"}})

    with pytest.raises(RuntimeError, match="rate limited"):
        TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())


def test_it_surfaces_the_server_message_from_an_http_error(tmp_path, monkeypatch):
    """Otherwise a 400 reads only as "HTTP Error 400: Bad Request"."""
    body = json.dumps(
        {"detail": {"error_type": "api_usage_error", "message": "Unknown model: jev-9"}}
    ).encode()

    def raise_http_error(request, timeout=None):
        raise urllib.error.HTTPError(
            ENDPOINT, 400, "Bad Request", {}, io.BytesIO(body)
        )

    monkeypatch.setattr(jev_module.urllib.request, "urlopen", raise_http_error)

    with pytest.raises(RuntimeError, match="Unknown model: jev-9"):
        TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())


def test_it_records_the_returned_answers_verbatim(tmp_path, fake_urlopen):
    fake_urlopen(response_body())

    result = TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())

    assert result.answers == {
        "f1_dependency": {"type": "noul", "noul": 0.95},
        "f2_dependency": {"type": "noul", "noul": 0.05},
    }


def test_it_records_only_the_answers_to_questions_it_asked(tmp_path, fake_urlopen):
    fake_urlopen(
        response_body(
            answers={
                "f1_dependency": {"type": "noul", "noul": 0.95},
                "f2_dependency": {"type": "noul", "noul": 0.05},
                "f9_dependency": {"type": "noul", "noul": 0.99},
            }
        )
    )

    result = TypeSafeJevRunner(make_config(tmp_path)).predict(make_instance())

    assert set(result.answers) == {"f1_dependency", "f2_dependency"}

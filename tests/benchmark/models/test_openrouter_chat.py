import json

import pytest

from benchmark.models.runners import openrouter_chat as openrouter_module
from benchmark.models.runners.base import Instance
from benchmark.models.runners.openrouter_chat import OpenRouterChatRunner

CONTENT = json.dumps({"dependencies": ["f1"], "justification": "because"})


def make_instance():
    return Instance(
        instance_id="webarena_000284_v1",
        scenario_id="webarena_000284",
        task="buy a thing",
        state=[{"id": "f1", "text": "a"}, {"id": "f2", "text": "b"}],
        proposed_action="Open the product page",
        supporting_fact_ids=["f1"],
    )


def make_config(tmp_path, **args):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("{{task}}\n{{state}}\n{{proposed_action}}")
    return {
        "name": "openrouter-test",
        "runner": "openrouter",
        "prompt": str(prompt),
        "args": {"model": "openai/gpt-oss-120b", **args},
    }


def response_body(content=CONTENT, **overrides):
    body = {
        "id": "gen-123",
        "model": "openai/gpt-oss-120b",
        "provider": "Fireworks",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 80,
            "prompt_tokens_details": {"cached_tokens": 1024},
        },
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
        monkeypatch.setattr(openrouter_module.urllib.request, "urlopen", fake)
        return fake

    return install


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


def sent_payload(fake):
    return json.loads(fake.request.data)


def test_it_refuses_to_build_without_an_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenRouterChatRunner(make_config(tmp_path))


def test_it_sends_the_key_as_a_bearer_token(tmp_path, fake_urlopen):
    fake = fake_urlopen(response_body())

    OpenRouterChatRunner(make_config(tmp_path)).predict(make_instance())

    assert fake.request.headers["Authorization"] == "Bearer sk-or-test"


def test_it_posts_the_rendered_prompt_to_the_chat_completions_endpoint(tmp_path, fake_urlopen):
    fake = fake_urlopen(response_body())

    OpenRouterChatRunner(make_config(tmp_path)).predict(make_instance())

    assert fake.request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    payload = sent_payload(fake)
    assert payload["model"] == "openai/gpt-oss-120b"
    assert payload["messages"] == [
        {"role": "user", "content": "buy a thing\nf1: a\nf2: b\nOpen the product page"}
    ]


def test_it_defaults_to_a_temperature_of_zero(tmp_path, fake_urlopen):
    fake = fake_urlopen(response_body())

    OpenRouterChatRunner(make_config(tmp_path)).predict(make_instance())

    assert sent_payload(fake)["temperature"] == 0


def test_it_sends_the_configured_params(tmp_path, fake_urlopen):
    fake = fake_urlopen(response_body())
    config = make_config(tmp_path, params={"temperature": 0.7, "max_tokens": 512})

    OpenRouterChatRunner(config).predict(make_instance())

    payload = sent_payload(fake)
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 512


def test_it_omits_reasoning_when_the_config_does_not_ask_for_it(tmp_path, fake_urlopen):
    fake = fake_urlopen(response_body())

    OpenRouterChatRunner(make_config(tmp_path)).predict(make_instance())

    assert "reasoning" not in sent_payload(fake)


def test_it_passes_reasoning_through_verbatim(tmp_path, fake_urlopen):
    fake = fake_urlopen(response_body())
    config = make_config(tmp_path, reasoning={"effort": "high"})

    OpenRouterChatRunner(config).predict(make_instance())

    assert sent_payload(fake)["reasoning"] == {"effort": "high"}


def test_it_honours_the_configured_timeout(tmp_path, fake_urlopen):
    fake = fake_urlopen(response_body())

    OpenRouterChatRunner(make_config(tmp_path, timeout_s=42)).predict(make_instance())

    assert fake.timeout == 42


def test_it_reads_the_dependencies_out_of_the_reply(tmp_path, fake_urlopen):
    fake_urlopen(response_body())

    result = OpenRouterChatRunner(make_config(tmp_path)).predict(make_instance())

    assert result.predicted_ids == ["f1"]
    assert result.expected_ids == ["f1"]
    assert result.meta["why"] == "because"


def test_it_parses_a_reply_wrapped_in_a_code_fence(tmp_path, fake_urlopen):
    fake_urlopen(response_body(content=f"```json\n{CONTENT}\n```"))

    result = OpenRouterChatRunner(make_config(tmp_path)).predict(make_instance())

    assert result.predicted_ids == ["f1"]


def test_it_maps_usage_onto_the_token_counts(tmp_path, fake_urlopen):
    fake_urlopen(response_body())

    result = OpenRouterChatRunner(make_config(tmp_path)).predict(make_instance())

    assert result.prompt_eval_count == 1200
    assert result.eval_count == 80
    assert result.prompt_eval_cached_count == 1024


def test_it_times_the_call_itself_since_openrouter_reports_no_durations(
    tmp_path, fake_urlopen, monkeypatch
):
    fake_urlopen(response_body())
    ticks = iter([100.0, 102.5])
    monkeypatch.setattr(openrouter_module.time, "monotonic", lambda: next(ticks))

    result = OpenRouterChatRunner(make_config(tmp_path)).predict(make_instance())

    assert result.total_duration_s == 2.5
    assert result.load_duration_s is None
    assert result.prompt_eval_duration_s is None
    assert result.eval_duration_s is None


def test_it_records_what_openrouter_actually_served(tmp_path, fake_urlopen):
    fake_urlopen(response_body())

    result = OpenRouterChatRunner(make_config(tmp_path)).predict(make_instance())

    assert result.meta["model"] == "openai/gpt-oss-120b"
    assert result.meta["provider"] == "Fireworks"
    assert result.meta["generation_id"] == "gen-123"


def test_it_keeps_the_reasoning_text_as_thinking(tmp_path, fake_urlopen):
    body = response_body()
    body["choices"][0]["message"]["reasoning"] = "let me think"
    fake_urlopen(body)

    result = OpenRouterChatRunner(make_config(tmp_path)).predict(make_instance())

    assert result.meta["thinking"] == "let me think"


def test_it_raises_when_a_200_reply_carries_an_error(tmp_path, fake_urlopen):
    fake_urlopen({"error": {"code": 402, "message": "insufficient credits"}})

    with pytest.raises(RuntimeError, match="insufficient credits"):
        OpenRouterChatRunner(make_config(tmp_path)).predict(make_instance())

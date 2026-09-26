from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from benchmark.models.runners.base import Instance, Runner, TaskResult, render_json

API_KEY_VAR = "TYPESAFE_API_KEY"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def fact_ids(instance: Instance) -> list[str]:
    """Every fact id in the state, in order, with nothing asked about twice.

    A duplicate id would collapse two questions into one key and under-ask the
    scenario without any error, so it is refused rather than tolerated.
    """
    ids = [fact["id"] for fact in instance.state]
    if not ids:
        raise ValueError(f"{instance.instance_id} has no observed facts to ask about")

    duplicates = sorted({fact_id for fact_id in ids if ids.count(fact_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate fact id(s) in state: {', '.join(duplicates)}")
    return ids


def _error_detail(http_error: urllib.error.HTTPError) -> str:
    """The server's own explanation, or the raw body when it is not JSON."""
    raw = http_error.read().decode(errors="replace")
    try:
        detail = json.loads(raw).get("detail", raw)
    except ValueError:
        return raw.strip()
    return detail.get("message", raw) if isinstance(detail, dict) else str(detail)


def state_lines(instance: Instance) -> list[str]:
    """One `id: text` line per fact - the dataset's `annotations` left behind."""
    return [f"{fact['id']}: {fact['text']}" for fact in instance.state]


def build_request(
    template: dict[str, Any], instance: Instance
) -> tuple[dict[str, Any], dict[str, str]]:
    """The `state` and `questions` blocks, and the question key -> fact id map.

    The state is rendered once; the question is expanded once per fact, so an
    N-fact scenario asks exactly N Nouls in the one request. Wording is
    identical across them but for the fact id.
    """
    state = render_json(
        template["state"],
        {
            "task": instance.task,
            "proposed_action": instance.proposed_action,
            "state_lines": state_lines(instance),
        },
    )

    ids = fact_ids(instance)
    keys = {render_json(template["question_key"], {"fact_id": fact_id}): fact_id for fact_id in ids}
    if len(keys) != len(ids):
        # a question_key that lost its {{fact_id}} would collapse every question
        # into one, asking about a single fact while the run looks healthy
        raise ValueError(
            f"question_key {template['question_key']!r} does not vary by fact: "
            f"{len(ids)} facts collapse to {len(keys)} question(s)"
        )
    questions = {
        key: render_json(template["question"], {"fact_id": fact_id})
        for key, fact_id in keys.items()
    }
    return {"state": state, "questions": questions}, keys


class TypeSafeJevRunner(Runner):
    """Jev, asked one Noul per fact against a single shared state.

    Unlike the generative conditions there is no prose prompt and no reply to
    parse: the scenario goes out as structured state, one question per fact
    comes back as a probability, and code thresholds those into the dependency
    set. Jev is never asked to explain itself, so `meta["why"]` stays unset.

    The result carries only fields every other runner carries, so a Jev record
    compares like for like against theirs.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.template = json.loads(Path(config["prompt"]).read_text())

        self.api_key = os.environ.get(API_KEY_VAR)
        if not self.api_key:
            raise RuntimeError(
                f"{API_KEY_VAR} is not set; export it or put it in the repo-root .env"
            )

        self.model = self.args["model"]
        self.threshold = self.args.get("threshold", 0.5)
        self.timeout_s = self.args.get("timeout_s", 120)

    def read_answers(self, body: dict[str, Any], keys: dict[str, str]) -> dict[str, Any]:
        """The answer to each question asked, verbatim and checked usable.

        Only the questions actually asked are read, so an answer to something
        never asked is ignored rather than thresholded into the prediction. A
        missing answer fails the instance instead of predicting a partial set.
        """
        returned = body.get("answers") or {}
        answers = {}
        for key, fact_id in keys.items():
            answer = returned.get(key)
            noul = answer.get("noul") if isinstance(answer, dict) else None
            # bool is an int in Python, and True >= 0.5 would read as certainty
            if isinstance(noul, bool) or not isinstance(noul, int | float):
                raise RuntimeError(
                    f"no usable noul answer for fact {fact_id} (question {key!r}): {answer!r}"
                )
            answers[key] = answer
        return answers

    def predict(self, instance: Instance) -> TaskResult:
        body, keys = build_request(self.template, instance)
        payload = {"model": self.model, **body}

        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                response_body = json.loads(response.read())
        except urllib.error.HTTPError as http_error:
            raise RuntimeError(
                f"typesafe HTTP {http_error.code}: {_error_detail(http_error)}"
            ) from http_error
        elapsed = time.monotonic() - started
        error = response_body.get("error")
        if error:
            raise RuntimeError(f"typesafe error {error.get('code')}: {error.get('message')}")

        answers = self.read_answers(response_body, keys)
        scores = {fact_id: answers[key]["noul"] for key, fact_id in keys.items()}
        usage = response_body.get("usage") or {}

        return TaskResult(
            predicted_ids=[
                fact_id for fact_id, score in scores.items() if score >= self.threshold
            ],
            expected_ids=instance.supporting_fact_ids,
            total_duration_s=elapsed,
            prompt_eval_count=usage.get("input_tokens"),
            eval_count=usage.get("output_tokens"),
            answers=answers,
            meta={
                "model": response_body.get("model"),
                "threshold": self.threshold,
            },
        )

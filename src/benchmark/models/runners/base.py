from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PLACEHOLDER_PATTERN = re.compile(r"\{\{(\w+)\}\}")


@dataclass
class Instance:
    """One scenario: what the user asked, what the agent saw, what it proposes to do."""

    instance_id: str
    task: str
    state: list[dict[str, Any]]
    proposed_action: str
    supporting_fact_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> Instance:
        data = json.loads(Path(path).read_text())
        return cls(
            instance_id=data["instance_id"],
            task=data["webarena_data"]["intent"],
            state=data["state"],
            proposed_action=data["proposed_action"]["text"],
            supporting_fact_ids=data.get("supporting_fact_ids", []),
        )

    def state_text(self) -> str:
        """The state as the prompt shows it: one `id: text` line per fact."""
        return "\n".join(f"{fact['id']}: {fact['text']}" for fact in self.state)


@dataclass
class TaskResult:
    """What every model needs to output for one instance."""

    predicted_ids: list[str] | None  # facts the model said the action depends on
    expected_ids: list[str] | None = None  # facts the task says it depends on
    total_duration_s: float | None = None  # the whole reply, load and prefill included
    load_duration_s: float | None = None  # getting the model into memory, ~0 once warm
    prompt_eval_count: int | None = None  # input tokens prefilled
    prompt_eval_cached_count: int | None = None  # how many of those were cache hits
    prompt_eval_duration_s: float | None = None  # time spent prefilling them
    eval_count: int | None = None  # output tokens generated, thinking included
    eval_duration_s: float | None = None  # time spent generating them
    error: str | None = None  # why this instance has no prediction
    precision: float = field(init=False, default=0.0)  # of the predicted, how many were right
    recall: float = field(init=False, default=0.0)  # of the expected, how many were found
    f1: float = field(init=False, default=0.0)  # the harmonic mean of the two
    meta: dict[str, Any] = field(default_factory=dict)  # anything else the runner keeps

    def __post_init__(self) -> None:
        if self.predicted_ids is not None:
            self.predicted_ids = sorted(self.predicted_ids)

        predicted = set(self.predicted_ids or [])
        expected = set(self.expected_ids or [])
        hits = len(predicted & expected)

        self.precision = round(hits / len(predicted), 4) if predicted else float(not expected)
        self.recall = round(hits / len(expected), 4) if expected else float(not predicted)
        self.f1 = (
            round(2 * self.precision * self.recall / (self.precision + self.recall), 4)
            if self.precision + self.recall
            else 0.0
        )


class Runner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.args = config.get("args", {})

    def render_prompt(self, template: str, instance: Any) -> str:
        """Fill a prompt template's `{{task}}`, `{{state}}`, `{{proposed_action}}`."""
        values = {
            "task": instance.task,
            "state": instance.state_text(),
            "proposed_action": instance.proposed_action,
        }
        rendered = PLACEHOLDER_PATTERN.sub(lambda m: values.get(m.group(1), m.group(0)), template)
        leftover = PLACEHOLDER_PATTERN.findall(rendered)
        if leftover:
            raise ValueError(
                f"template has unfilled placeholder(s): {', '.join(sorted(set(leftover)))}"
            )
        return rendered

    def predict(self, instance: Instance) -> TaskResult:
        raise NotImplementedError("predict must be implemented by the runner")

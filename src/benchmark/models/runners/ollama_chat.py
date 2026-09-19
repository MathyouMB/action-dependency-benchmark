from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from benchmark.models.runners.base import Instance, Runner, TaskResult


class OllamaChatRunner(Runner):
    """Shared /api/chat plumbing for Ollama-backed runners.

    Subclasses handle whatever is specific to their model family (how `think`
    is expressed, its allowed values/default, quirks in parsing, ...). This
    base class only knows how to call Ollama and read a JSON-object reply out
    of `message.content`.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.template = Path(config["prompt"]).read_text()

        self.model = self.args["model"]
        self.host = self.args.get("host", "http://localhost:11434").rstrip("/")
        self.timeout_s = self.args.get("timeout_s", 600)
        self.think = self.build_think()

    def build_think(self) -> Any | None:
        """The `think` value to send, or None to omit it. Override for a
        model family that supports/needs it."""
        return None

    def parse_content(self, content: str) -> dict[str, Any]:
        """Parse the model's reply into the expected JSON object. Override if
        a model family doesn't return bare JSON (e.g. wraps it in a markdown
        code fence)."""
        return json.loads(content)

    def predict(self, instance: Instance) -> TaskResult:
        prompt = self.render_prompt(self.template, instance)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": self.args.get("options", {"temperature": 0}),
        }
        if self.think is not None:
            payload["think"] = self.think

        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            response_body = json.loads(response.read())

        message = response_body.get("message", {})
        parsed = self.parse_content(message.get("content", "{}"))

        total_duration_ns = response_body.get("total_duration")

        return TaskResult(
            predicted_ids=parsed.get("dependencies", []),
            expected_ids=instance.supporting_fact_ids,
            total_duration_s=total_duration_ns / 1e9 if total_duration_ns is not None else None,
            meta={
                "why": parsed.get("justification"),
                "thinking": message.get("thinking"),
                "eval_count": response_body.get("eval_count"),
                "prompt_eval_count": response_body.get("prompt_eval_count"),
                "think": self.think,
                "model": self.model,
            },
        )

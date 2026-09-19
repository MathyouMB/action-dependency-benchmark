from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from benchmark.models.runners.base import Instance, Runner, TaskResult

THINK_LEVELS = ("low", "medium", "high")


class GptOssOllamaRunner(Runner):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.template = Path(config["prompt"]).read_text()

        self.model = self.args.get("model", "gpt-oss:20b")
        self.host = self.args.get("host", "http://localhost:11434").rstrip("/")
        self.timeout_s = self.args.get("timeout_s", 600)

        self.think = self.args.get("think", "medium")
        if self.think not in THINK_LEVELS:
            raise ValueError(f"`think` must be one of {THINK_LEVELS}, got {self.think!r}")

    def predict(self, instance: Instance) -> TaskResult:
        prompt = self.render_prompt(self.template, instance)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "think": self.think,
            "stream": False,
            "options": self.args.get("options", {"temperature": 0}),
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            response_body = json.loads(response.read())

        message = response_body.get("message", {})
        parsed = json.loads(message.get("content", "{}"))

        total_duration_ns = response_body.get("total_duration")

        return TaskResult(
            predicted_ids=parsed.get("dependencies", []),
            expected_ids=instance.supporting_fact_ids,
            total_duration_s=total_duration_ns / 1e9 if total_duration_ns is not None else None,
            meta={
                "thinking": message.get("thinking"),
                "eval_count": response_body.get("eval_count"),
                "prompt_eval_count": response_body.get("prompt_eval_count"),
                "think": self.think,
                "model": self.model,
            },
        )

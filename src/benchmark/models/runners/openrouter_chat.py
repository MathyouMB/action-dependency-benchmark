from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from benchmark.models.runners.base import CODE_FENCE_PATTERN, Instance, Runner, TaskResult

API_KEY_VAR = "OPENROUTER_API_KEY"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterChatRunner(Runner):
    """One runner for every model OpenRouter serves.

    OpenRouter normalizes the chat-completions API across providers, so
    unlike the Ollama side there is no per-family subclass: `reasoning` is
    passed through as the config writes it, and replies are parsed
    fence-tolerantly because any family may wrap its JSON in one.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.template = Path(config["prompt"]).read_text()

        self.api_key = os.environ.get(API_KEY_VAR)
        if not self.api_key:
            raise RuntimeError(
                f"{API_KEY_VAR} is not set; export it or put it in the repo-root .env"
            )

        self.model = self.args["model"]
        self.timeout_s = self.args.get("timeout_s", 600)
        self.params = self.args.get("params", {"temperature": 0})
        self.reasoning = self.args.get("reasoning")

    def parse_content(self, content: str) -> dict[str, Any]:
        match = CODE_FENCE_PATTERN.match(content)
        return json.loads(match.group(1) if match else content)

    def predict(self, instance: Instance) -> TaskResult:
        prompt = self.render_prompt(self.template, instance)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **self.params,
        }
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning

        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            response_body = json.loads(response.read())
        elapsed = time.monotonic() - started

        # OpenRouter reports some upstream failures in a 200 body, not an HTTP error.
        error = response_body.get("error")
        if error:
            raise RuntimeError(f"openrouter error {error.get('code')}: {error.get('message')}")

        message = response_body["choices"][0].get("message", {})
        parsed = self.parse_content(message.get("content", "{}"))
        usage = response_body.get("usage") or {}

        return TaskResult(
            predicted_ids=parsed.get("dependencies", []),
            expected_ids=instance.supporting_fact_ids,
            total_duration_s=elapsed,
            prompt_eval_count=usage.get("prompt_tokens"),
            prompt_eval_cached_count=(usage.get("prompt_tokens_details") or {}).get(
                "cached_tokens"
            ),
            eval_count=usage.get("completion_tokens"),
            meta={
                "why": parsed.get("justification"),
                "thinking": message.get("reasoning"),
                "reasoning": self.reasoning,
                # what OpenRouter actually served, which can differ from what we asked for
                "model": response_body.get("model"),
                "provider": response_body.get("provider"),
                "generation_id": response_body.get("id"),
            },
        )

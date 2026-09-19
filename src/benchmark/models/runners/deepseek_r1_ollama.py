from __future__ import annotations

import json
import re
from typing import Any

from benchmark.models.runners.ollama_chat import OllamaChatRunner

CODE_FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class DeepSeekR1OllamaRunner(OllamaChatRunner):
    """DeepSeek-R1 takes `think` as a plain bool (on by default it's a
    reasoning-only model). It also tends to wrap its JSON answer in a
    ```json ... ``` code fence regardless of instructions not to."""

    def build_think(self) -> bool:
        return bool(self.args.get("think", True))

    def parse_content(self, content: str) -> dict[str, Any]:
        match = CODE_FENCE_PATTERN.match(content)
        return json.loads(match.group(1) if match else content)

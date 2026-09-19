from __future__ import annotations

from typing import Any

from benchmark.models.runners.ollama_chat import OllamaChatRunner

THINK_LEVELS = ("low", "medium", "high")


class GptOssOllamaRunner(OllamaChatRunner):
    """gpt-oss models take `think` as a graded effort level, not a bool."""

    def build_think(self) -> Any:
        think = self.args.get("think", "medium")
        if think not in THINK_LEVELS:
            raise ValueError(f"`think` must be one of {THINK_LEVELS}, got {think!r}")
        return think

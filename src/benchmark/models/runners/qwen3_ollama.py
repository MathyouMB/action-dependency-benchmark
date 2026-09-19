from __future__ import annotations

from benchmark.models.runners.ollama_chat import OllamaChatRunner


class Qwen3OllamaRunner(OllamaChatRunner):
    """Qwen3 is a hybrid model: `think` is a plain bool, off by default."""

    def build_think(self) -> bool:
        return bool(self.args.get("think", False))

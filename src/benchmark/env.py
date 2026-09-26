from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


def load_env_file(path: str | Path = DEFAULT_ENV_FILE) -> None:
    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)

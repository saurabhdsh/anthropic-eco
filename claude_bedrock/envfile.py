"""Load a local .env without overriding variables the instance already has."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    file = Path(path)
    if not file.is_file():
        return
    for raw in file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def setting(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None

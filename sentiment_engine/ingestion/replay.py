"""Replay helpers for JSONL event files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {file_path}:{line_no}") from exc

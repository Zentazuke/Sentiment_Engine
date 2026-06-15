"""Text normalization helpers."""

from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    cleaned = _URL_RE.sub(" ", text)
    cleaned = cleaned.replace("$", " ")
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned.strip().lower()

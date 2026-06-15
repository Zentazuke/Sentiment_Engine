"""Sentiment scoring dispatcher.

Default: the fast rule-based lexicon (processing/sentiment_fast.py) — cheap,
offline, transparent, and what the engine has always used.

If SENTIMENT_TRANSFORMER_MODEL is set, each text is scored by that transformer
(CryptoBERT/FinBERT) instead, with an automatic fall-back to the lexicon if the
model is unavailable or inference fails. The public contract is identical to the
lexicon's, so callers don't change:

    score_text(text) -> (sentiment Decimal in [-1, 1], confidence Decimal in [0, 1])
"""

from __future__ import annotations

from decimal import Decimal
from typing import Tuple

from sentiment_engine.processing.sentiment_fast import score_text as _lexicon_score_text
from sentiment_engine.processing.sentiment_onnx import onnx_enabled, score_with_onnx
from sentiment_engine.processing.sentiment_transformer import (
    score_with_transformer,
    transformer_enabled,
)


def score_text(text: str) -> Tuple[Decimal, Decimal]:
    # Priority: lightweight ONNX-int8 -> PyTorch transformer -> rule-based lexicon.
    if onnx_enabled():
        result = score_with_onnx(text)
        if result is not None:
            return result
    if transformer_enabled():
        result = score_with_transformer(text)
        if result is not None:
            return result
    return _lexicon_score_text(text)

"""Optional transformer sentiment scorer (CryptoBERT / FinBERT).

Off by default. Set SENTIMENT_TRANSFORMER_MODEL to a Hugging Face model id
(e.g. "ElKulako/cryptobert" for social, "ProsusAI/finbert" for news) to enable
it. Requires `pip install transformers torch` on the machine that runs the
engine; if those aren't installed, or the model can't load, scoring falls back
to the rule-based lexicon (see processing/scoring.py) and the engine keeps
working unchanged.

The label->score mapping is a pure function so it's fully unit-testable without
the heavy ML dependencies.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import List, Optional, Tuple

from sentiment_engine.config import TRANSFORMER_MODEL

logger = logging.getLogger(__name__)

_BULLISH = {"bullish", "positive", "label_2", "pos"}
_BEARISH = {"bearish", "negative", "label_0", "neg"}

_pipeline = None
_load_failed = False


def transformer_enabled() -> bool:
    return bool(TRANSFORMER_MODEL)


def score_from_distribution(scores: List[dict]) -> Tuple[float, float]:
    """[{label, score}, ...] -> (sentiment in [-1,1], confidence in [0,1]).

    sentiment = P(bullish) - P(bearish); confidence = peak class probability.
    """
    p_bull = 0.0
    p_bear = 0.0
    peak = 0.0
    for entry in scores:
        label = str(entry.get("label", "")).strip().lower()
        prob = float(entry.get("score", 0.0))
        peak = max(peak, prob)
        if label in _BULLISH:
            p_bull += prob
        elif label in _BEARISH:
            p_bear += prob
    sentiment = max(-1.0, min(1.0, p_bull - p_bear))
    return sentiment, max(0.0, min(1.0, peak))


def _get_pipeline():
    """Lazy-load the HF text-classification pipeline once. None if unavailable."""
    global _pipeline, _load_failed
    if _pipeline is not None or _load_failed:
        return _pipeline
    try:
        from transformers import pipeline  # heavy import, only when enabled
        _pipeline = pipeline(
            "text-classification", model=TRANSFORMER_MODEL, top_k=None, truncation=True, max_length=128,
        )
        logger.info("transformer scorer: loaded %s", TRANSFORMER_MODEL)
    except Exception as exc:  # noqa: BLE001 - any failure -> fall back to lexicon
        _load_failed = True
        logger.warning("transformer scorer: could not load %s (%s); using lexicon",
                       TRANSFORMER_MODEL, type(exc).__name__)
    return _pipeline


def score_with_transformer(text: str) -> Optional[Tuple[Decimal, Decimal]]:
    """(sentiment, confidence) Decimals, or None if the transformer is unavailable."""
    pipe = _get_pipeline()
    if pipe is None:
        return None
    try:
        result = pipe(text)
        # pipeline(top_k=None) returns a list of {label,score}; batched call nests one level.
        scores = result[0] if result and isinstance(result[0], list) else result
        sentiment, confidence = score_from_distribution(scores)
        return Decimal(str(round(sentiment, 4))), Decimal(str(round(confidence, 4)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("transformer scorer: inference failed (%s); using lexicon", type(exc).__name__)
        return None

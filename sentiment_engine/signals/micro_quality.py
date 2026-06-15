"""Quality scoring for microstructure data, separate from social data quality.

A scalping decision should be gated by how trustworthy the MARKET data is,
not by how many tweets exist. Missing/stale microstructure scores 0 and the
decision layer falls back to sentiment-only behavior (which itself leans
neutral without data).
"""

from __future__ import annotations

from typing import Optional

from sentiment_engine.config import MICRO_MIN_TRADES_60S
from sentiment_engine.types import MicrostructureSnapshot

# Presence weights for the metrics the decision layer actually uses.
_PRESENCE_WEIGHTS = (
    ("trade_imbalance_10s", 0.25),
    ("trade_imbalance_60s", 0.15),
    ("book_imbalance", 0.20),
    ("spread_bps", 0.15),
)
_ACTIVITY_WEIGHT = 0.25  # trade_count_60s vs MICRO_MIN_TRADES_60S


def micro_quality_score(micro: Optional[MicrostructureSnapshot], stale: bool) -> float:
    """0..1 quality of microstructure data. Stale or missing data scores 0."""
    if micro is None or stale:
        return 0.0
    score = sum(weight for field, weight in _PRESENCE_WEIGHTS if getattr(micro, field) is not None)
    count = micro.trade_count_60s or 0
    if MICRO_MIN_TRADES_60S > 0:
        score += _ACTIVITY_WEIGHT * min(1.0, count / MICRO_MIN_TRADES_60S)
    else:
        score += _ACTIVITY_WEIGHT
    return round(min(1.0, score), 4)


def micro_direction_score(micro: MicrostructureSnapshot) -> Optional[float]:
    """Directional pressure in [-1, 1]; positive = buy pressure.

    Weighted blend of trade-flow and book imbalance, re-weighted over the
    metrics that are actually available. Requires at least one trade-flow
    imbalance; book imbalance alone is too easy to spoof.
    """
    components = (
        (micro.trade_imbalance_10s, 0.40),
        (micro.trade_imbalance_30s, 0.25),
        (micro.book_imbalance, 0.35),
    )
    if micro.trade_imbalance_10s is None and micro.trade_imbalance_30s is None:
        return None
    total_weight = sum(weight for value, weight in components if value is not None)
    if total_weight <= 0:
        return None
    score = sum(value * weight for value, weight in components if value is not None) / total_weight
    return round(max(-1.0, min(1.0, score)), 4)

"""Attention spike scoring."""

from __future__ import annotations

from decimal import Decimal


def attention_spike_score(current_count: int, previous_count: int) -> Decimal:
    """Normalize message-count acceleration to a 0..1 score.

    A value near 0 means no unusual attention. A value near 1 means current
    discussion volume is much higher than the previous comparable window.
    """
    if current_count <= 0:
        return Decimal("0")
    baseline = max(previous_count, 1)
    ratio = Decimal(current_count) / Decimal(baseline)
    # ratio 1.0 -> 0.0, ratio 3.0+ -> 1.0
    normalized = (ratio - Decimal("1")) / Decimal("2")
    if normalized < 0:
        normalized = Decimal("0")
    if normalized > 1:
        normalized = Decimal("1")
    return normalized.quantize(Decimal("0.0001"))

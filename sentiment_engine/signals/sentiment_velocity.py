"""Sentiment velocity calculation."""

from __future__ import annotations

from decimal import Decimal


def sentiment_velocity_score(current_average: Decimal, previous_average: Decimal) -> Decimal:
    return (current_average - previous_average).quantize(Decimal("0.0001"))

"""Price/sentiment alignment helpers."""

from __future__ import annotations

from decimal import Decimal

from sentiment_engine.types import TradeDirection


def price_change_pct(first_price: Decimal | None, last_price: Decimal | None) -> Decimal:
    if first_price is None or last_price is None or first_price <= 0:
        return Decimal("0")
    return (((last_price - first_price) / first_price) * Decimal("100")).quantize(Decimal("0.0001"))


def aligns_with_direction(direction: TradeDirection, sentiment_velocity: Decimal, price_change: Decimal) -> bool:
    if direction is TradeDirection.LONG:
        return sentiment_velocity > 0 and price_change >= 0
    return sentiment_velocity < 0 and price_change <= 0


def contradicts_direction(direction: TradeDirection, sentiment_velocity: Decimal) -> bool:
    if direction is TradeDirection.LONG:
        return sentiment_velocity < 0
    return sentiment_velocity > 0

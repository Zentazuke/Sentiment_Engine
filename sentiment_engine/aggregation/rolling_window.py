"""In-memory rolling event windows."""

from __future__ import annotations

import time
from collections import deque
from decimal import Decimal
from typing import Deque, Iterable, List, Tuple

from sentiment_engine.config import MAX_EVENT_AGE_SECONDS
from sentiment_engine.types import PriceEvent, SocialEvent


class RollingWindow:
    """Stores recent social and price events for one symbol."""

    def __init__(self) -> None:
        self.social_events: Deque[SocialEvent] = deque()
        self.price_events: Deque[PriceEvent] = deque()

    def add_social(self, event: SocialEvent) -> None:
        self.social_events.append(event)
        self.prune()

    def add_price(self, event: PriceEvent) -> None:
        self.price_events.append(event)
        self.prune()

    def prune(self, now: float | None = None) -> None:
        cutoff = (now if now is not None else time.time()) - MAX_EVENT_AGE_SECONDS
        while self.social_events and self.social_events[0].timestamp < cutoff:
            self.social_events.popleft()
        while self.price_events and self.price_events[0].timestamp < cutoff:
            self.price_events.popleft()

    def social_between(self, start: float, end: float) -> List[SocialEvent]:
        return [event for event in self.social_events if start <= event.timestamp < end]

    def prices_between(self, start: float, end: float) -> List[PriceEvent]:
        return [event for event in self.price_events if start <= event.timestamp <= end]


def average_sentiment(events: Iterable[SocialEvent]) -> Decimal:
    values = [event.sentiment for event in events if event.sentiment is not None]
    if not values:
        return Decimal("0")
    return (sum(values, Decimal("0")) / Decimal(len(values))).quantize(Decimal("0.0001"))


def first_last_price(events: Iterable[PriceEvent]) -> Tuple[Decimal | None, Decimal | None]:
    ordered = sorted(events, key=lambda event: event.timestamp)
    if not ordered:
        return None, None
    return ordered[0].price, ordered[-1].price

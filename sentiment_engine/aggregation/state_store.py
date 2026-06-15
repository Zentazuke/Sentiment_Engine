"""Shared in-memory state for the standalone sentiment engine."""

from __future__ import annotations

import threading
import time
from collections import deque
from decimal import Decimal
from typing import Deque, Dict, List, Optional, Tuple

from sentiment_engine.aggregation.rolling_window import RollingWindow, average_sentiment, first_last_price
from sentiment_engine.config import (
    CONTEXT_STALE_SECONDS,
    MAIN_WINDOW_SECONDS,
    MICRO_STALE_SECONDS,
    PREVIOUS_WINDOW_SECONDS,
    SUPPORTED_SYMBOLS,
)
from sentiment_engine.processing.coin_mapper import normalize_symbol
from sentiment_engine.signals.attention import attention_spike_score
from sentiment_engine.signals.data_quality import data_quality_score
from sentiment_engine.signals.price_divergence import price_change_pct
from sentiment_engine.signals.sentiment_velocity import sentiment_velocity_score
from sentiment_engine.types import ContextSnapshot, MicrostructureSnapshot, PriceEvent, SentimentSnapshot, SocialEvent


class StateStore:
    """Thread-safe in-memory state for recent social and price events."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._windows: Dict[str, RollingWindow] = {symbol: RollingWindow() for symbol in SUPPORTED_SYMBOLS}
        # Latest microstructure snapshot per symbol + when we received it.
        self._micro: Dict[str, Tuple[MicrostructureSnapshot, float]] = {}
        # Small display buffer: most recent social events per symbol,
        # kept regardless of window age (news is sparse).
        self._recent_social: Dict[str, Deque[SocialEvent]] = {}
        # Latest global market context (one per engine, not per symbol).
        self._context: Optional[Tuple[ContextSnapshot, float]] = None

    def add_social(self, event: SocialEvent) -> None:
        symbol = normalize_symbol(event.symbol)
        with self._lock:
            self._windows.setdefault(symbol, RollingWindow()).add_social(event)
            self._recent_social.setdefault(symbol, deque(maxlen=50)).append(event)

    def recent_social(self, symbol: str, limit: int = 20) -> List[SocialEvent]:
        """Most recent social events (newest first) for display purposes."""
        normalized = normalize_symbol(symbol)
        with self._lock:
            events = list(self._recent_social.get(normalized, ()))
        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]

    def add_price(self, event: PriceEvent) -> None:
        symbol = normalize_symbol(event.symbol)
        with self._lock:
            self._windows.setdefault(symbol, RollingWindow()).add_price(event)

    def add_context(self, snapshot: ContextSnapshot, received_at: float | None = None) -> None:
        with self._lock:
            self._context = (snapshot, received_at if received_at is not None else time.time())

    def context(self, now: float | None = None) -> Tuple[Optional[ContextSnapshot], bool]:
        """Latest context snapshot and whether it is stale (>1h old)."""
        current = now if now is not None else time.time()
        with self._lock:
            entry = self._context
        if entry is None:
            return None, True
        snapshot, received_at = entry
        return snapshot, (current - received_at) > CONTEXT_STALE_SECONDS

    def latest_price(self, symbol: str) -> Optional[float]:
        """Most recent ingested price for a symbol, or None."""
        normalized = normalize_symbol(symbol)
        with self._lock:
            window = self._windows.get(normalized)
            if window is None or not window.price_events:
                return None
            return float(window.price_events[-1].price)

    def add_microstructure(self, snapshot: MicrostructureSnapshot, received_at: float | None = None) -> None:
        symbol = normalize_symbol(snapshot.symbol)
        with self._lock:
            self._micro[symbol] = (snapshot, received_at if received_at is not None else time.time())

    def microstructure(self, symbol: str, now: float | None = None) -> Tuple[Optional[MicrostructureSnapshot], bool]:
        """Latest microstructure snapshot and whether it is stale.

        Returns (None, True) when nothing has been ingested yet. Stale data
        must never be used to confirm or veto - callers lean neutral.
        """
        normalized = normalize_symbol(symbol)
        current = now if now is not None else time.time()
        with self._lock:
            entry = self._micro.get(normalized)
        if entry is None:
            return None, True
        snapshot, received_at = entry
        # Conservative: stale if EITHER computed long ago or received long ago.
        age = current - min(received_at, snapshot.computed_at)
        return snapshot, age > MICRO_STALE_SECONDS

    def snapshot(self, symbol: str, now: float | None = None) -> SentimentSnapshot:
        normalized = normalize_symbol(symbol)
        current_time = now if now is not None else time.time()
        current_start = current_time - MAIN_WINDOW_SECONDS
        previous_start = current_start - PREVIOUS_WINDOW_SECONDS

        with self._lock:
            window = self._windows.setdefault(normalized, RollingWindow())
            window.prune(current_time)
            current_social = window.social_between(current_start, current_time)
            previous_social = window.social_between(previous_start, current_start)
            current_prices = window.prices_between(current_start, current_time)

        current_avg = average_sentiment(current_social)
        previous_avg = average_sentiment(previous_social)
        velocity = sentiment_velocity_score(current_avg, previous_avg)
        attention = attention_spike_score(len(current_social), len(previous_social))
        first_price, last_price = first_last_price(current_prices)
        price_pct = price_change_pct(first_price, last_price)
        newest_ts = max((event.timestamp for event in current_social), default=None)
        sources = [event.source for event in current_social]
        quality = data_quality_score(
            message_count=len(current_social),
            sources=sources,
            newest_timestamp=newest_ts,
        )

        return SentimentSnapshot(
            symbol=normalized,
            window_seconds=MAIN_WINDOW_SECONDS,
            message_count=len(current_social),
            previous_message_count=len(previous_social),
            average_sentiment=current_avg,
            previous_average_sentiment=previous_avg,
            sentiment_velocity=velocity,
            attention_spike=attention,
            price_change_pct=price_pct,
            data_quality=quality,
        )


STATE = StateStore()

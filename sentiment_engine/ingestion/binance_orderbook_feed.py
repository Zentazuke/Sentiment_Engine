"""Binance partial order book WebSocket feed (public market data, read-only).

Subscribes to <symbol>@depth20@100ms and keeps only the latest book per
symbol. Book metrics are computed via the pure functions in
``sentiment_engine.signals.microstructure``.

This module never places orders and never authenticates.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from sentiment_engine.config import BINANCE_WS_URL, MAX_WS_SESSION_SECONDS, MICRO_DEPTH_BAND_PCT
from sentiment_engine.processing.coin_mapper import normalize_symbol
from sentiment_engine.signals.microstructure import (
    BookLevel,
    book_imbalance,
    depth_quote_value,
    spread_bps,
)

logger = logging.getLogger(__name__)

_RECONNECT_MIN_SECONDS = 1.0
_RECONNECT_MAX_SECONDS = 30.0


@dataclass(slots=True)
class BookState:
    bids: List[BookLevel] = field(default_factory=list)  # sorted desc by price
    asks: List[BookLevel] = field(default_factory=list)  # sorted asc by price
    updated_at: float = 0.0


def parse_levels(raw_levels: list) -> List[BookLevel]:
    return [(float(price), float(qty)) for price, qty in raw_levels]


class BinanceOrderBookFeed:
    """Keeps the latest 20-level book per symbol from Binance depth stream."""

    def __init__(self, symbols: Sequence[str]) -> None:
        self.symbols = [normalize_symbol(s) for s in symbols]
        self._stream_to_symbol = {
            f"{normalize_symbol(s).replace('/', '').lower()}@depth20@100ms": s
            for s in self.symbols
        }
        self._books: Dict[str, BookState] = {s: BookState() for s in self.symbols}
        self.connected: bool = False

    # --- data collection -------------------------------------------------

    def handle_message(self, raw: str) -> None:
        """Parse a combined-stream depth message and store it. Never raises."""
        try:
            message = json.loads(raw)
            symbol = self._stream_to_symbol.get(message.get("stream", ""))
            if symbol is None:
                return
            data = message["data"]
            book = self._books[symbol]
            book.bids = parse_levels(data["bids"])
            book.asks = parse_levels(data["asks"])
            book.updated_at = time.time()
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("book feed: unparseable message (%s)", exc)

    async def run(self) -> None:
        """Connect and consume forever, reconnecting with backoff on failure."""
        import websockets  # local import: keeps unit tests dependency-free

        streams = "/".join(self._stream_to_symbol)
        url = f"{BINANCE_WS_URL}?streams={streams}"
        backoff = _RECONNECT_MIN_SECONDS
        while True:
            try:
                logger.info("book feed: connecting %s", url)
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    self.connected = True
                    backoff = _RECONNECT_MIN_SECONDS
                    logger.info("book feed: connected (%s)", ", ".join(self.symbols))
                    session_started = time.time()
                    async for raw in ws:
                        self.handle_message(raw)
                        if time.time() - session_started > MAX_WS_SESSION_SECONDS:
                            logger.info("book feed: recycling session before exchange forced disconnect")
                            break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect boundary
                logger.warning("book feed: disconnected (%s: %s); retry in %.1fs",
                               type(exc).__name__, exc, backoff)
            self.connected = False
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_SECONDS)

    # --- metrics ----------------------------------------------------------

    def metrics(self, symbol: str, now: Optional[float] = None, max_age: float = 5.0) -> dict:
        """Compute book metrics for one symbol; empty dict if book missing/old."""
        current = now if now is not None else time.time()
        book = self._books.get(normalize_symbol(symbol))
        if book is None or not book.bids or not book.asks:
            return {}
        if current - book.updated_at > max_age:
            return {}  # old book is worse than no book

        best_bid = book.bids[0][0]
        best_ask = book.asks[0][0]
        spread = spread_bps(best_bid, best_ask)
        if spread is None:
            return {}  # crossed/garbled book - refuse to report any of it
        mid = (best_bid + best_ask) / 2.0
        bid_value = depth_quote_value(book.bids, mid, MICRO_DEPTH_BAND_PCT)
        ask_value = depth_quote_value(book.asks, mid, MICRO_DEPTH_BAND_PCT)
        return {
            "spread_bps": spread,
            "bid_depth_quote": bid_value,
            "ask_depth_quote": ask_value,
            "book_imbalance": book_imbalance(bid_value, ask_value),
        }

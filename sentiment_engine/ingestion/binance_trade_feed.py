"""Binance aggTrade WebSocket feed (public market data, read-only).

Collects recent trades per symbol and computes trade-flow metrics via the
pure functions in ``sentiment_engine.signals.microstructure``.

This module never places orders, never authenticates, and never generates
trade signals. It only observes the public trade stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Deque, Dict, Optional, Sequence

from sentiment_engine.config import (
    MAX_WS_SESSION_SECONDS,
    BINANCE_WS_URL,
    MICRO_BASELINE_MIN_SECONDS,
    MICRO_BASELINE_SECONDS,
    MICRO_VWAP_WINDOW_SECONDS,
)
from sentiment_engine.processing.coin_mapper import normalize_symbol
from sentiment_engine.signals.microstructure import (
    Trade,
    buy_sell_volume,
    momentum_pct,
    relative_volume,
    trade_imbalance,
    trades_in_window,
    volatility_bps,
    vwap,
    vwap_distance_bps,
)

logger = logging.getLogger(__name__)

_RECONNECT_MIN_SECONDS = 1.0
_RECONNECT_MAX_SECONDS = 30.0


def binance_stream_symbol(symbol: str) -> str:
    """'BTC/USDT' -> 'btcusdt'."""
    return normalize_symbol(symbol).replace("/", "").lower()


def parse_agg_trade(data: dict) -> Trade:
    """Parse one Binance aggTrade payload into a Trade.

    Binance fields: p=price, q=quantity, T=trade time (ms),
    m=True when the buyer is the maker (i.e. the taker SOLD).
    """
    return Trade(
        price=float(data["p"]),
        quantity=float(data["q"]),
        timestamp=float(data["T"]) / 1000.0,
        is_aggressive_buy=not bool(data["m"]),
    )


class BinanceTradeFeed:
    """Maintains a rolling trade buffer per symbol from Binance aggTrade."""

    def __init__(self, symbols: Sequence[str]) -> None:
        self.symbols = [normalize_symbol(s) for s in symbols]
        self._stream_to_symbol = {
            f"{binance_stream_symbol(s)}@aggTrade": s for s in self.symbols
        }
        self._buffers: Dict[str, Deque[Trade]] = {s: deque() for s in self.symbols}
        self.last_message_at: Optional[float] = None
        self.connected: bool = False

    # --- data collection -------------------------------------------------

    def handle_message(self, raw: str) -> None:
        """Parse a combined-stream message and store the trade. Never raises."""
        try:
            message = json.loads(raw)
            stream = message.get("stream", "")
            symbol = self._stream_to_symbol.get(stream)
            if symbol is None:
                return
            trade = parse_agg_trade(message["data"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("trade feed: unparseable message (%s)", exc)
            return
        buffer = self._buffers[symbol]
        buffer.append(trade)
        self.last_message_at = time.time()
        self._prune(buffer)

    def _prune(self, buffer: Deque[Trade], now: Optional[float] = None) -> None:
        cutoff = (now if now is not None else time.time()) - MICRO_BASELINE_SECONDS
        while buffer and buffer[0].timestamp < cutoff:
            buffer.popleft()

    async def run(self) -> None:
        """Connect and consume forever, reconnecting with backoff on failure."""
        import websockets  # local import: keeps unit tests dependency-free

        streams = "/".join(self._stream_to_symbol)
        url = f"{BINANCE_WS_URL}?streams={streams}"
        backoff = _RECONNECT_MIN_SECONDS
        while True:
            try:
                logger.info("trade feed: connecting %s", url)
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    self.connected = True
                    backoff = _RECONNECT_MIN_SECONDS
                    logger.info("trade feed: connected (%s)", ", ".join(self.symbols))
                    session_started = time.time()
                    async for raw in ws:
                        self.handle_message(raw)
                        if time.time() - session_started > MAX_WS_SESSION_SECONDS:
                            logger.info("trade feed: recycling session before exchange forced disconnect")
                            break
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect boundary
                logger.warning("trade feed: disconnected (%s: %s); retry in %.1fs",
                               type(exc).__name__, exc, backoff)
            self.connected = False
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_MAX_SECONDS)

    # --- metrics ----------------------------------------------------------

    def metrics(self, symbol: str, now: Optional[float] = None) -> dict:
        """Compute trade-flow metrics for one symbol. Missing data -> None values."""
        current = now if now is not None else time.time()
        buffer = self._buffers.get(normalize_symbol(symbol))
        if not buffer:
            return {}

        trades_10s = trades_in_window(buffer, current - 10, current)
        trades_30s = trades_in_window(buffer, current - 30, current)
        trades_60s = trades_in_window(buffer, current - 60, current)
        trades_vwap = trades_in_window(buffer, current - MICRO_VWAP_WINDOW_SECONDS, current)

        buy_10s, sell_10s = buy_sell_volume(trades_10s)
        buy_30s, sell_30s = buy_sell_volume(trades_30s)
        buy_60s, sell_60s = buy_sell_volume(trades_60s)

        last_price = buffer[-1].price if buffer else None

        # Relative volume: last 60s vs trailing per-60s baseline (after warmup).
        rel_volume = None
        buffer_age = current - buffer[0].timestamp
        if buffer_age >= MICRO_BASELINE_MIN_SECONDS:
            total_volume = sum(t.quantity for t in buffer)
            baseline_per_60s = total_volume * 60.0 / buffer_age
            rel_volume = relative_volume(buy_60s + sell_60s, baseline_per_60s)

        return {
            "last_price": last_price,
            "buy_volume_10s": buy_10s if trades_10s else None,
            "sell_volume_10s": sell_10s if trades_10s else None,
            "trade_imbalance_10s": trade_imbalance(buy_10s, sell_10s),
            "trade_imbalance_30s": trade_imbalance(buy_30s, sell_30s),
            "trade_imbalance_60s": trade_imbalance(buy_60s, sell_60s),
            "relative_volume": rel_volume,
            "momentum_pct_30s": momentum_pct(trades_30s),
            "volatility_bps_30s": volatility_bps(trades_30s),
            "vwap_distance_bps": vwap_distance_bps(last_price or 0.0, vwap(trades_vwap)),
            "trade_count_60s": len(trades_60s) if trades_60s else None,
        }

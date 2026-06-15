"""Pure microstructure metric functions.

Every function here is side-effect free and operates on plain inputs so it can
be unit tested without any network connection. The live Binance feeds collect
raw trades/books and call these functions; nothing in this module talks to an
exchange.

Conventions:
- Metrics use ``float`` (these are fast statistical signals, not monetary
  amounts that need ``Decimal`` precision).
- Imbalance values are in [-1, 1]; positive means buy/bid pressure.
- ``*_bps`` values are basis points (1 bp = 0.01%).
- Functions return ``None`` when there is not enough data. Callers must treat
  ``None`` as "metric unavailable" and lean neutral - never as zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True, slots=True)
class Trade:
    """One aggregated trade from an exchange feed."""

    price: float
    quantity: float
    timestamp: float  # unix seconds
    is_aggressive_buy: bool  # taker bought (buyer was NOT the maker)


BookLevel = Tuple[float, float]  # (price, quantity)


def trades_in_window(trades: Sequence[Trade], start: float, end: float) -> List[Trade]:
    """Trades with start < timestamp <= end."""
    return [t for t in trades if start < t.timestamp <= end]


def buy_sell_volume(trades: Sequence[Trade]) -> Tuple[float, float]:
    """Aggressive (taker) buy and sell base volume."""
    buy = sum(t.quantity for t in trades if t.is_aggressive_buy)
    sell = sum(t.quantity for t in trades if not t.is_aggressive_buy)
    return buy, sell


def trade_imbalance(buy_volume: float, sell_volume: float) -> Optional[float]:
    """(buy - sell) / (buy + sell) in [-1, 1]. None when there is no volume."""
    total = buy_volume + sell_volume
    if total <= 0:
        return None
    return (buy_volume - sell_volume) / total


def relative_volume(window_volume: float, baseline_volume_per_window: float) -> Optional[float]:
    """Current window volume vs a trailing baseline for the same window size.

    1.0 = average activity, 3.0 = three times the usual volume.
    """
    if baseline_volume_per_window <= 0:
        return None
    return window_volume / baseline_volume_per_window


def momentum_pct(trades: Sequence[Trade]) -> Optional[float]:
    """Percent price change from first to last trade in the window."""
    if len(trades) < 2:
        return None
    first = trades[0].price
    last = trades[-1].price
    if first <= 0:
        return None
    return (last - first) / first * 100.0


def volatility_bps(trades: Sequence[Trade], bucket_seconds: float = 1.0) -> Optional[float]:
    """Std-dev of per-bucket returns, in basis points.

    Trades are bucketed by time (last price per bucket) so bursts of trades do
    not inflate volatility purely through trade count.
    """
    if len(trades) < 4 or bucket_seconds <= 0:
        return None
    buckets: Dict[int, float] = {}
    for trade in trades:  # assumed time-ordered; last write wins per bucket
        buckets[int(trade.timestamp // bucket_seconds)] = trade.price
    prices = [buckets[key] for key in sorted(buckets)]
    returns = [(b - a) / a for a, b in zip(prices, prices[1:]) if a > 0]
    if len(returns) < 3:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    return math.sqrt(variance) * 10_000.0


def vwap(trades: Sequence[Trade]) -> Optional[float]:
    """Volume-weighted average price over the given trades."""
    total_quantity = sum(t.quantity for t in trades)
    if total_quantity <= 0:
        return None
    return sum(t.price * t.quantity for t in trades) / total_quantity


def vwap_distance_bps(price: float, vwap_price: Optional[float]) -> Optional[float]:
    """How far price sits from VWAP, in bps. Positive = above VWAP."""
    if vwap_price is None or vwap_price <= 0 or price <= 0:
        return None
    return (price - vwap_price) / vwap_price * 10_000.0


def spread_bps(best_bid: Optional[float], best_ask: Optional[float]) -> Optional[float]:
    """Bid/ask spread in bps of the mid price."""
    if best_bid is None or best_ask is None or best_bid <= 0 or best_ask <= 0:
        return None
    if best_ask < best_bid:  # crossed/garbled book - refuse to score
        return None
    mid = (best_bid + best_ask) / 2.0
    return (best_ask - best_bid) / mid * 10_000.0


def depth_quote_value(levels: Sequence[BookLevel], mid_price: float, band_pct: float) -> Optional[float]:
    """Sum of price*qty (quote value) for levels within +-band_pct% of mid."""
    if mid_price <= 0 or band_pct <= 0:
        return None
    band = mid_price * band_pct / 100.0
    return sum(p * q for p, q in levels if abs(p - mid_price) <= band and p > 0 and q > 0)


def book_imbalance(bid_value: Optional[float], ask_value: Optional[float]) -> Optional[float]:
    """(bid - ask) / (bid + ask) over near-price depth, in [-1, 1]."""
    if bid_value is None or ask_value is None:
        return None
    total = bid_value + ask_value
    if total <= 0:
        return None
    return (bid_value - ask_value) / total

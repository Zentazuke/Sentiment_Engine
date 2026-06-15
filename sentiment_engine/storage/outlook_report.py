"""Outlook validation report.

Answers the only question that matters about "predictability": when the
outlook said bullish/bearish, what did price actually do afterwards?

Realized returns come from micro_history (the live feed records last_price
every second), looked up at outlook time + horizon with a tolerance.

Run after the system has been live for a while:
    python -m sentiment_engine.storage.outlook_report --symbol BTC/USDT --days 7
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sentiment_engine.config import JOURNAL_DB_PATH
from sentiment_engine.processing.coin_mapper import normalize_symbol

# Score magnitudes below this are treated as "no call" and excluded from hit-rate.
NEUTRAL_BAND = 0.12
# A price is usable if found within this many seconds of the target time.
PRICE_TOLERANCE_SECONDS = 600.0

# (computed_at, score, confidence, price_at_compute)
OutlookRow = Tuple[float, float, float, Optional[float]]
PriceLookup = Callable[[float], Optional[float]]


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def evaluate_horizon(
    outlooks: Sequence[OutlookRow],
    price_at: PriceLookup,
    horizon_seconds: float,
) -> Dict[str, object]:
    """Pure: hit-rate, mean returns, and correlation for one horizon."""
    scores: List[float] = []
    returns: List[float] = []
    hits = 0
    calls = 0
    bullish_returns: List[float] = []
    bearish_returns: List[float] = []
    for computed_at, score, _confidence, price_then in outlooks:
        if price_then is None or price_then <= 0:
            continue
        price_later = price_at(computed_at + horizon_seconds)
        if price_later is None or price_later <= 0:
            continue
        realized = (price_later - price_then) / price_then * 100.0
        scores.append(score)
        returns.append(realized)
        if abs(score) >= NEUTRAL_BAND:
            calls += 1
            if (score > 0 and realized > 0) or (score < 0 and realized < 0):
                hits += 1
            (bullish_returns if score > 0 else bearish_returns).append(realized)
    return {
        "samples": len(returns),
        "directional_calls": calls,
        "hit_rate": round(hits / calls, 4) if calls else None,
        "mean_return_when_bullish_pct": round(sum(bullish_returns) / len(bullish_returns), 4) if bullish_returns else None,
        "mean_return_when_bearish_pct": round(sum(bearish_returns) / len(bearish_returns), 4) if bearish_returns else None,
        "correlation": round(pearson(scores, returns), 4) if pearson(scores, returns) is not None else None,
    }


def make_price_lookup(conn: sqlite3.Connection, symbol: str) -> PriceLookup:
    def lookup(target_ts: float) -> Optional[float]:
        row = conn.execute(
            "SELECT last_price FROM micro_history WHERE symbol = ? AND last_price IS NOT NULL"
            " AND received_at BETWEEN ? AND ? ORDER BY ABS(received_at - ?) LIMIT 1",
            (symbol, target_ts - PRICE_TOLERANCE_SECONDS, target_ts + PRICE_TOLERANCE_SECONDS, target_ts),
        ).fetchone()
        return float(row[0]) if row else None
    return lookup


def print_report(db_path: str, symbol: str, days: float) -> None:
    normalized = normalize_symbol(symbol)
    cutoff = time.time() - days * 86400.0
    conn = sqlite3.connect(db_path)
    try:
        outlooks: List[OutlookRow] = [
            (float(c), float(s), float(conf), float(p) if p is not None else None)
            for c, s, conf, p in conn.execute(
                "SELECT computed_at, composite_score, confidence, price_at_compute"
                " FROM outlook_history WHERE symbol = ? AND computed_at >= ? ORDER BY computed_at",
                (normalized, cutoff),
            ).fetchall()
        ]
        lookup = make_price_lookup(conn, normalized)
        print(f"\n{normalized} outlook validation - last {days:g} days - {len(outlooks)} outlooks\n")
        if not outlooks:
            print("No outlooks journaled yet. Let the dashboard/engine run with auto-refresh on.")
            return
        header = f"{'horizon':<10}{'samples':>9}{'calls':>7}{'hit rate':>10}{'bull ret%':>11}{'bear ret%':>11}{'corr':>8}"
        print(header)
        print("-" * len(header))
        for label, seconds in (("1h", 3600), ("6h", 21600), ("24h", 86400)):
            r = evaluate_horizon(outlooks, lookup, seconds)
            def fmt(v, spec=".4f"):
                return format(v, spec) if v is not None else "-"
            print(f"{label:<10}{r['samples']:>9}{r['directional_calls']:>7}{fmt(r['hit_rate']):>10}"
                  f"{fmt(r['mean_return_when_bullish_pct']):>11}{fmt(r['mean_return_when_bearish_pct']):>11}"
                  f"{fmt(r['correlation']):>8}")
        print("\nRead with care: hit rates near 0.5 mean no edge; few samples mean noise.\n")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Outlook vs realized-return validation report.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument("--db", default=JOURNAL_DB_PATH)
    args = parser.parse_args()
    print_report(args.db, args.symbol, args.days)


if __name__ == "__main__":
    main()

"""Calibration stats over recorded microstructure history.

Run after letting run_live record for a while:
    python -m sentiment_engine.storage.micro_stats --symbol BTC/USDT --hours 24

Prints count + percentiles per metric so confirm/veto/gate thresholds in
config can be set from real distributions.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from typing import List, Optional, Sequence

from sentiment_engine.config import JOURNAL_DB_PATH
from sentiment_engine.processing.coin_mapper import normalize_symbol
from sentiment_engine.storage.micro_history import METRIC_COLUMNS

_PERCENTILES = (5, 25, 50, 75, 95, 99)


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of pre-sorted values."""
    if not sorted_values:
        raise ValueError("no values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def metric_summary(values: List[float]) -> Optional[dict]:
    present = sorted(v for v in values if v is not None)
    if not present:
        return None
    return {
        "count": len(present),
        "coverage": len(present) / len(values) if values else 0.0,
        **{f"p{p}": percentile(present, p) for p in _PERCENTILES},
    }


def print_stats(db_path: str, symbol: str, hours: float) -> None:
    normalized = normalize_symbol(symbol)
    cutoff = time.time() - hours * 3600.0
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT {', '.join(METRIC_COLUMNS)} FROM micro_history"
            " WHERE symbol = ? AND received_at >= ? ORDER BY received_at",
            (normalized, cutoff),
        ).fetchall()
    finally:
        conn.close()

    print(f"\n{normalized} - last {hours:g}h - {len(rows)} snapshots\n")
    if not rows:
        print("No history recorded yet. Run run_live against the engine first.")
        return
    header = f"{'metric':<22}{'n':>7}{'cov%':>6}" + "".join(f"{f'p{p}':>11}" for p in _PERCENTILES)
    print(header)
    print("-" * len(header))
    for index, name in enumerate(METRIC_COLUMNS):
        summary = metric_summary([row[index] for row in rows])
        if summary is None:
            print(f"{name:<22}{'0':>7}{'0':>6}" + "       (never reported)")
            continue
        line = f"{name:<22}{summary['count']:>7}{summary['coverage'] * 100:>5.0f}%"
        line += "".join(f"{summary[f'p{p}']:>11.4f}" for p in _PERCENTILES)
        print(line)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Microstructure calibration stats.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--db", default=JOURNAL_DB_PATH)
    args = parser.parse_args()
    print_stats(args.db, args.symbol, args.hours)


if __name__ == "__main__":
    main()

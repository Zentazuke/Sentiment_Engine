"""Persistent store for Binance long/short positioning ratios.

Keyless Binance futures "long/short ratio" data is crowd-positioning, not text
sentiment: the share of accounts/positions that are long vs short, plus taker
buy/sell pressure. We journal it per symbol so the calibration learner can test
whether it leads price (and in which direction) the same way it tests sentiment.

One wide row per poll per symbol. `signal()` maps a ratio's long-fraction to a
[-1, 1] crowd-net-long signal (positive = net long).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from sentiment_engine.config import JOURNAL_DB_PATH


def long_fraction_signal(long_fraction: Optional[float]) -> Optional[float]:
    """0..1 long share -> [-1, 1] (0.5 -> 0, all-long -> +1, all-short -> -1)."""
    if long_fraction is None:
        return None
    return max(-1.0, min(1.0, (long_fraction - 0.5) * 2.0))


def ratio_to_long_fraction(long_short_ratio: Optional[float]) -> Optional[float]:
    """longShortRatio (longs/shorts) -> long fraction longs/(longs+shorts)."""
    if long_short_ratio is None or long_short_ratio < 0:
        return None
    return long_short_ratio / (1.0 + long_short_ratio)


class PositioningHistory:
    def __init__(self, path: str | Path = JOURNAL_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positioning_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                symbol TEXT NOT NULL,
                global_account_ratio REAL,
                top_account_ratio REAL,
                top_position_ratio REAL,
                taker_ratio REAL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_positioning_symbol_time"
            " ON positioning_history (symbol, timestamp)"
        )
        self._conn.commit()

    def record(
        self,
        *,
        symbol: str,
        global_account_ratio: Optional[float],
        top_account_ratio: Optional[float],
        top_position_ratio: Optional[float],
        taker_ratio: Optional[float],
        timestamp: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO positioning_history"
                " (timestamp, symbol, global_account_ratio, top_account_ratio,"
                "  top_position_ratio, taker_ratio) VALUES (?, ?, ?, ?, ?, ?)",
                (timestamp if timestamp is not None else time.time(), symbol,
                 global_account_ratio, top_account_ratio, top_position_ratio, taker_ratio),
            )
            self._conn.commit()

    def latest(self, symbol: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT timestamp, global_account_ratio, top_account_ratio,"
                " top_position_ratio, taker_ratio FROM positioning_history"
                " WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        ts, glob, top_acct, top_pos, taker = row
        return {
            "timestamp": ts,
            "global_account_ratio": glob,
            "top_account_ratio": top_acct,
            "top_position_ratio": top_pos,
            "taker_ratio": taker,
            # learner-facing signal: crowd net-long from the global account ratio
            "signal": long_fraction_signal(ratio_to_long_fraction(glob)),
        }

    def series_between(self, symbol: str, start: float, end: float) -> List[Tuple[float, float]]:
        """(timestamp, global crowd-net-long signal) within [start, end]."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT timestamp, global_account_ratio FROM positioning_history"
                " WHERE symbol = ? AND timestamp > ? AND timestamp <= ? ORDER BY timestamp",
                (symbol, start, end),
            ).fetchall()
        out: List[Tuple[float, float]] = []
        for ts, glob in rows:
            sig = long_fraction_signal(ratio_to_long_fraction(glob))
            if sig is not None:
                out.append((ts, sig))
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_singleton: Optional[PositioningHistory] = None


def get_positioning_history() -> PositioningHistory:
    global _singleton
    if _singleton is None:
        _singleton = PositioningHistory()
    return _singleton

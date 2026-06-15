"""Persistent store for Binance derivatives signals: funding rate + open interest.

Both are keyless Binance USD-M futures data and complement the long/short ratio:

- Funding rate: the periodic fee longs pay shorts (or vice-versa). Persistently
  positive funding = crowded, over-leveraged longs (a classic contrarian tell);
  negative = crowded shorts. Already a small SIGNED number, so it doubles as a
  signal directly.
- Open interest (USD notional): how much money is in open positions. Its level is
  not directional, but its CHANGE is: rising OI behind a price move = real new
  conviction; falling OI = unwinding / short-covering.

One wide row per poll per symbol. `funding_signal()` maps a raw funding rate to a
bounded [-1, 1] value (positive = longs paying = net-long crowding).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from sentiment_engine.config import JOURNAL_DB_PATH

# Funding rate magnitude treated as "extreme" when normalizing to [-1, 1].
# Typical 8h funding is ~0.01%; 0.05% per interval is already very crowded.
_FUNDING_EXTREME = 0.0005


def funding_signal(funding_rate: Optional[float]) -> Optional[float]:
    """Raw 8h funding rate -> bounded [-1, 1] crowding signal."""
    if funding_rate is None:
        return None
    return max(-1.0, min(1.0, funding_rate / _FUNDING_EXTREME))


class DerivativesHistory:
    def __init__(self, path: str | Path = JOURNAL_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS derivatives_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                symbol TEXT NOT NULL,
                funding_rate REAL,
                open_interest_usd REAL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_derivatives_symbol_time"
            " ON derivatives_history (symbol, timestamp)"
        )
        self._conn.commit()

    def record(
        self,
        *,
        symbol: str,
        funding_rate: Optional[float],
        open_interest_usd: Optional[float],
        timestamp: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO derivatives_history"
                " (timestamp, symbol, funding_rate, open_interest_usd) VALUES (?, ?, ?, ?)",
                (timestamp if timestamp is not None else time.time(), symbol,
                 funding_rate, open_interest_usd),
            )
            self._conn.commit()

    def latest(self, symbol: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT timestamp, funding_rate, open_interest_usd FROM derivatives_history"
                " WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        ts, funding, oi = row
        return {
            "timestamp": ts,
            "funding_rate": funding,
            "open_interest_usd": oi,
            "funding_signal": funding_signal(funding),
        }

    def funding_series_between(self, symbol: str, start: float, end: float) -> List[Tuple[float, float]]:
        """(timestamp, funding signal) within [start, end] for the learner."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT timestamp, funding_rate FROM derivatives_history"
                " WHERE symbol = ? AND timestamp > ? AND timestamp <= ? ORDER BY timestamp",
                (symbol, start, end),
            ).fetchall()
        out: List[Tuple[float, float]] = []
        for ts, funding in rows:
            sig = funding_signal(funding)
            if sig is not None:
                out.append((ts, sig))
        return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_singleton: Optional[DerivativesHistory] = None


def get_derivatives_history() -> DerivativesHistory:
    global _singleton
    if _singleton is None:
        _singleton = DerivativesHistory()
    return _singleton

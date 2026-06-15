"""SQLite recorder for microstructure snapshots.

Every snapshot pushed to /ingest/microstructure is appended here so that
confirm/veto thresholds can be calibrated from real distributions instead of
guesses. Old rows are pruned periodically (retention is configurable).

Calibration stats:
    python -m sentiment_engine.storage.micro_stats --symbol BTC/USDT --hours 24
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Final, Optional

from sentiment_engine.config import JOURNAL_DB_PATH, MICRO_HISTORY_RETENTION_DAYS
from sentiment_engine.types import MicrostructureSnapshot

METRIC_COLUMNS: Final = (
    "last_price",
    "buy_volume_10s",
    "sell_volume_10s",
    "trade_imbalance_10s",
    "trade_imbalance_30s",
    "trade_imbalance_60s",
    "relative_volume",
    "momentum_pct_30s",
    "volatility_bps_30s",
    "vwap_distance_bps",
    "bid_depth_quote",
    "ask_depth_quote",
    "book_imbalance",
    "spread_bps",
    "trade_count_60s",
)

_PRUNE_EVERY_INSERTS = 1000


class MicroHistory:
    """Append-only history of microstructure snapshots with retention pruning."""

    def __init__(self, path: str | Path = JOURNAL_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        columns = ",\n                ".join(f"{name} REAL" for name in METRIC_COLUMNS)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS micro_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at REAL NOT NULL,
                computed_at REAL NOT NULL,
                symbol TEXT NOT NULL,
                {columns}
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_micro_history_symbol_time"
            " ON micro_history (symbol, received_at)"
        )
        self._conn.commit()
        self._inserts_since_prune = 0

    def record(self, snapshot: MicrostructureSnapshot, received_at: float | None = None) -> None:
        values = [
            received_at if received_at is not None else time.time(),
            snapshot.computed_at,
            snapshot.symbol,
        ] + [getattr(snapshot, name) for name in METRIC_COLUMNS]
        placeholders = ", ".join("?" for _ in values)
        column_names = "received_at, computed_at, symbol, " + ", ".join(METRIC_COLUMNS)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO micro_history ({column_names}) VALUES ({placeholders})", values
            )
            self._conn.commit()
            self._inserts_since_prune += 1
            if self._inserts_since_prune >= _PRUNE_EVERY_INSERTS:
                self._prune_locked()
                self._inserts_since_prune = 0

    def _prune_locked(self) -> None:
        cutoff = time.time() - MICRO_HISTORY_RETENTION_DAYS * 86400.0
        self._conn.execute("DELETE FROM micro_history WHERE received_at < ?", (cutoff,))
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_MICRO_HISTORY: Optional[MicroHistory] = None


def get_micro_history() -> MicroHistory:
    """Lazy singleton so importing this module never opens the database."""
    global _MICRO_HISTORY
    if _MICRO_HISTORY is None:
        _MICRO_HISTORY = MicroHistory()
    return _MICRO_HISTORY

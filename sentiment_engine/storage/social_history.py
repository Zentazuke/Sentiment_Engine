"""Persistent storage for social/news events and computed outlooks.

The in-memory rolling windows only cover minutes (scalping layer). Medium-term
outlooks need hours-to-days of events that survive restarts, so every ingested
social event is also appended here. Computed outlooks are journaled too, which
makes the outlook score backtestable against subsequent price moves.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

from sentiment_engine.config import JOURNAL_DB_PATH, SOCIAL_HISTORY_RETENTION_DAYS, SYNTHETIC_SOURCES
from sentiment_engine.types import SocialEvent

_PRUNE_EVERY_INSERTS = 500

# (timestamp, source, sentiment) - all an outlook needs per event.
EventRow = Tuple[float, str, Optional[float]]


class SocialHistory:
    def __init__(self, path: str | Path = JOURNAL_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS social_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                symbol TEXT NOT NULL,
                source TEXT NOT NULL,
                author TEXT,
                text TEXT NOT NULL,
                sentiment REAL,
                confidence REAL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_social_history_symbol_time"
            " ON social_history (symbol, timestamp)"
        )
        # Dedupe guard: identical (symbol, source, text, timestamp) rows are
        # ignored, which makes history backfills safe to re-run.
        try:
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_social_history_unique"
                " ON social_history (symbol, source, text, timestamp)"
            )
        except sqlite3.OperationalError:
            pass  # pre-existing duplicates; dedupe applies to new rows only
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS outlook_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                computed_at REAL NOT NULL,
                symbol TEXT NOT NULL,
                composite_score REAL NOT NULL,
                confidence REAL NOT NULL,
                context_tilt REAL NOT NULL,
                horizons_json TEXT NOT NULL,
                price_at_compute REAL
            )
            """
        )
        self._conn.commit()
        self._inserts = 0

    def record_event(self, event: SocialEvent) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO social_history (timestamp, symbol, source, author, text, sentiment, confidence)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    event.timestamp,
                    event.symbol,
                    event.source,
                    event.author,
                    event.text,
                    float(event.sentiment) if event.sentiment is not None else None,
                    float(event.confidence) if event.confidence is not None else None,
                ),
            )
            self._conn.commit()
            self._inserts += 1
            if self._inserts >= _PRUNE_EVERY_INSERTS:
                self._prune_locked()
                self._inserts = 0

    def events_between(
        self, symbol: str, start: float, end: float, *, exclude_synthetic: bool = False
    ) -> List[EventRow]:
        query = (
            "SELECT timestamp, source, sentiment FROM social_history"
            " WHERE symbol = ? AND timestamp > ? AND timestamp <= ?"
        )
        params: list = [symbol, start, end]
        if exclude_synthetic and SYNTHETIC_SOURCES:
            placeholders = ", ".join("?" for _ in SYNTHETIC_SOURCES)
            query += f" AND source NOT IN ({placeholders})"
            params.extend(SYNTHETIC_SOURCES)
        query += " ORDER BY timestamp"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [(float(ts), str(src), float(s) if s is not None else None) for ts, src, s in rows]

    def recent_events(self, symbol: str, limit: int = 20) -> List[Tuple[float, str, Optional[str], str, Optional[float]]]:
        """Newest-first (timestamp, source, author, text, sentiment) for display."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT timestamp, source, author, text, sentiment FROM social_history"
                " WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        return [(float(ts), str(src), a, str(txt), float(s) if s is not None else None)
                for ts, src, a, txt, s in rows]

    def text_rows_between(
        self, symbol: str, start: float, end: float, *, exclude_synthetic: bool = True
    ) -> List[Tuple[float, str, str, Optional[float]]]:
        """(timestamp, source, text, sentiment) rows for insights."""
        query = (
            "SELECT timestamp, source, text, sentiment FROM social_history"
            " WHERE symbol = ? AND timestamp > ? AND timestamp <= ?"
        )
        params: list = [symbol, start, end]
        if exclude_synthetic and SYNTHETIC_SOURCES:
            placeholders = ", ".join("?" for _ in SYNTHETIC_SOURCES)
            query += f" AND source NOT IN ({placeholders})"
            params.extend(SYNTHETIC_SOURCES)
        query += " ORDER BY timestamp"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [(float(ts), str(src), str(txt), float(s) if s is not None else None)
                for ts, src, txt, s in rows]

    def outlook_rows_since(self, symbol: str, since: float) -> List[Tuple[float, float, float, Optional[float]]]:
        """(computed_at, score, confidence, price_at_compute) for the timeline."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT computed_at, composite_score, confidence, price_at_compute"
                " FROM outlook_history WHERE symbol = ? AND computed_at >= ? ORDER BY computed_at",
                (symbol, since),
            ).fetchall()
        return [(float(c), float(s), float(conf), float(p) if p is not None else None)
                for c, s, conf, p in rows]

    def record_outlook(
        self,
        *,
        symbol: str,
        computed_at: float,
        composite_score: float,
        confidence: float,
        context_tilt: float,
        horizons_json: str,
        price_at_compute: Optional[float],
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO outlook_history (computed_at, symbol, composite_score, confidence,"
                " context_tilt, horizons_json, price_at_compute) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (computed_at, symbol, composite_score, confidence, context_tilt, horizons_json, price_at_compute),
            )
            self._conn.commit()

    def _prune_locked(self) -> None:
        cutoff = time.time() - SOCIAL_HISTORY_RETENTION_DAYS * 86400.0
        self._conn.execute("DELETE FROM social_history WHERE timestamp < ?", (cutoff,))
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


_SOCIAL_HISTORY: Optional[SocialHistory] = None


def get_social_history() -> SocialHistory:
    """Lazy singleton; importing this module never opens the database."""
    global _SOCIAL_HISTORY
    if _SOCIAL_HISTORY is None:
        _SOCIAL_HISTORY = SocialHistory()
    return _SOCIAL_HISTORY

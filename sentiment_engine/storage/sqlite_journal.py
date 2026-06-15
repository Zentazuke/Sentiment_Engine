"""SQLite journal for sentiment evaluations."""

from __future__ import annotations

import sqlite3
import threading
import time
from decimal import Decimal
from pathlib import Path

from sentiment_engine.config import JOURNAL_DB_PATH, OUTCOME_HORIZONS_SECONDS
from sentiment_engine.types import TradeDirection, TradeEvaluation


class SentimentJournal:
    def __init__(self, path: str | Path = JOURNAL_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                bot_confidence REAL NOT NULL,
                action TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                confidence_modifier REAL NOT NULL,
                final_confidence_estimate REAL NOT NULL,
                data_quality REAL NOT NULL,
                message_count INTEGER NOT NULL,
                sentiment_velocity REAL NOT NULL,
                attention_spike REAL NOT NULL,
                price_change_pct REAL NOT NULL,
                safe_to_use INTEGER NOT NULL,
                reason TEXT NOT NULL,
                price_at_eval REAL
            )
            """
        )
        self._outcome_columns = tuple(f"price_after_{h}s" for h in OUTCOME_HORIZONS_SECONDS)
        # Migrate older databases in place (ALTER is a no-op error if present).
        for column in ("price_at_eval",) + self._outcome_columns:
            try:
                self._conn.execute(f"ALTER TABLE evaluations ADD COLUMN {column} REAL")
            except sqlite3.OperationalError:
                pass  # column already exists
        self._conn.commit()

    def log_evaluation(
        self,
        *,
        evaluation: TradeEvaluation,
        direction: TradeDirection,
        bot_confidence: Decimal,
        timestamp: float | None = None,
        price_at_eval: float | None = None,
    ) -> int:
        """Insert one evaluation row and return its row id (for outcome fills)."""
        snap = evaluation.snapshot
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO evaluations (
                    timestamp, symbol, direction, bot_confidence, action,
                    sentiment_score, confidence_modifier, final_confidence_estimate,
                    data_quality, message_count, sentiment_velocity, attention_spike,
                    price_change_pct, safe_to_use, reason, price_at_eval
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp if timestamp is not None else time.time(),
                    evaluation.symbol,
                    direction.value,
                    float(bot_confidence),
                    evaluation.action.value,
                    float(evaluation.sentiment_score),
                    float(evaluation.confidence_modifier),
                    float(evaluation.final_confidence_estimate),
                    float(evaluation.data_quality),
                    snap.message_count,
                    float(snap.sentiment_velocity),
                    float(snap.attention_spike),
                    float(snap.price_change_pct),
                    1 if evaluation.safe_to_use else 0,
                    evaluation.reason,
                    price_at_eval,
                ),
            )
            row_id = int(cursor.lastrowid or 0)
            self._conn.commit()
            return row_id

    def fill_outcome(self, row_id: int, horizon_seconds: int, price: float | None) -> None:
        """Record the observed price N seconds after an evaluation."""
        column = f"price_after_{horizon_seconds}s"
        if column not in self._outcome_columns:
            raise ValueError(f"Unknown outcome horizon {horizon_seconds}s")
        with self._lock:
            self._conn.execute(
                f"UPDATE evaluations SET {column} = ? WHERE id = ?", (price, row_id)
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


JOURNAL = SentimentJournal()

"""Background outcome tracking for evaluations.

After every /evaluate call, this tracker records the observed price at each
configured horizon (default 30s/120s/300s) into the journal row. That makes
shadow-mode measurable: "would this confirm/veto have helped?"

Implementation: one daemon thread polling a due-list. If no live price is
available when a horizon comes due, the outcome stays NULL - never guessed.
This module reads prices; it never trades.
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from sentiment_engine.config import OUTCOME_HORIZONS_SECONDS

logger = logging.getLogger(__name__)

_POLL_SECONDS = 2.0
# A price is only a valid outcome if observed within this slack of the horizon.
_MAX_FILL_DELAY_SECONDS = 15.0

PriceGetter = Callable[[str], Optional[float]]
OutcomeWriter = Callable[[int, int, Optional[float]], None]


@dataclass(order=True)
class _PendingOutcome:
    due_at: float
    row_id: int = field(compare=False)
    symbol: str = field(compare=False)
    horizon_seconds: int = field(compare=False)


class OutcomeTracker:
    """Fills journal outcome columns as horizons come due."""

    def __init__(self, write_outcome: OutcomeWriter, get_price: PriceGetter) -> None:
        self._write_outcome = write_outcome
        self._get_price = get_price
        self._lock = threading.Lock()
        self._heap: List[_PendingOutcome] = []
        self._thread: Optional[threading.Thread] = None

    def register(self, row_id: int, symbol: str, evaluated_at: float) -> None:
        """Schedule outcome fills for one evaluation row."""
        with self._lock:
            for horizon in OUTCOME_HORIZONS_SECONDS:
                heapq.heappush(
                    self._heap,
                    _PendingOutcome(evaluated_at + horizon, row_id, symbol, horizon),
                )
        self._ensure_thread()

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="outcome-tracker", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        logger.info("outcome tracker started (horizons=%s)", OUTCOME_HORIZONS_SECONDS)
        while True:
            time.sleep(_POLL_SECONDS)
            self.process_due()

    def process_due(self, now: Optional[float] = None) -> int:
        """Fill all due outcomes; returns how many were processed. Testable."""
        current = now if now is not None else time.time()
        processed = 0
        while True:
            with self._lock:
                if not self._heap or self._heap[0].due_at > current:
                    break
                pending = heapq.heappop(self._heap)
            price: Optional[float] = None
            if current - pending.due_at <= _MAX_FILL_DELAY_SECONDS:
                try:
                    price = self._get_price(pending.symbol)
                except Exception as exc:  # noqa: BLE001 - tracker must not die
                    logger.warning("outcome price lookup failed: %s", exc)
            else:
                logger.warning(
                    "outcome for row %s (+%ss) filled too late; recording NULL",
                    pending.row_id, pending.horizon_seconds,
                )
            try:
                self._write_outcome(pending.row_id, pending.horizon_seconds, price)
                processed += 1
            except Exception as exc:  # noqa: BLE001 - tracker must not die
                logger.warning("outcome write failed for row %s: %s", pending.row_id, exc)
        return processed

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._heap)

"""Data quality scoring.

This keeps the engine humble. Low sample size or stale data returns neutral.
"""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Iterable

from sentiment_engine.config import MAIN_WINDOW_SECONDS, MIN_MESSAGES_FOR_SIGNAL


def data_quality_score(
    *,
    message_count: int,
    sources: Iterable[str],
    newest_timestamp: float | None,
) -> Decimal:
    if message_count <= 0 or newest_timestamp is None:
        return Decimal("0")

    sample_score = min(Decimal("1"), Decimal(message_count) / Decimal(MIN_MESSAGES_FOR_SIGNAL))
    source_count = len({s for s in sources if s})
    # Manual testing usually has one source; it can still reach 0.8 if sample/recency are good.
    source_score = min(Decimal("1"), Decimal("0.50") + Decimal(source_count) * Decimal("0.25"))

    age_seconds = max(0.0, time.time() - newest_timestamp)
    if age_seconds <= MAIN_WINDOW_SECONDS:
        recency_score = Decimal("1")
    elif age_seconds <= MAIN_WINDOW_SECONDS * 2:
        recency_score = Decimal("0.50")
    else:
        recency_score = Decimal("0")

    quality = (sample_score * Decimal("0.55")) + (source_score * Decimal("0.20")) + (recency_score * Decimal("0.25"))
    return max(Decimal("0"), min(Decimal("1"), quality)).quantize(Decimal("0.0001"))

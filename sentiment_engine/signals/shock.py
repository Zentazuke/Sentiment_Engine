"""Shock detection: sudden bursts of unusually negative/positive chatter.

A shock fires when BOTH conditions hold over the last ~10 minutes:
1. Message rate is a multiple of the trailing-hour baseline (attention burst).
2. Mean sentiment is decisively negative (panic) or positive (euphoria).

This is the part of social data with real short-term value: detecting that
something just happened. It is descriptive - it never says buy or sell.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from sentiment_engine.config import (
    SHOCK_BASELINE_SECONDS,
    SHOCK_MIN_EVENTS,
    SHOCK_RATE_RATIO,
    SHOCK_SENTIMENT_ABS,
    SHOCK_WINDOW_SECONDS,
)

EventRow = Tuple[float, str, Optional[float]]  # (timestamp, source, sentiment)


def detect_shock(rows: Sequence[EventRow], now: float) -> Dict[str, object]:
    """Pure shock assessment from event rows covering window+baseline."""
    window_start = now - SHOCK_WINDOW_SECONDS
    baseline_start = window_start - SHOCK_BASELINE_SECONDS

    window = [row for row in rows if window_start < row[0] <= now]
    baseline = [row for row in rows if baseline_start < row[0] <= window_start]

    window_rate = len(window) / SHOCK_WINDOW_SECONDS
    baseline_rate = len(baseline) / SHOCK_BASELINE_SECONDS
    # With no baseline activity, require the minimum event count to carry it.
    rate_ratio = (window_rate / baseline_rate) if baseline_rate > 0 else (
        float(len(window)) if window else 0.0
    )

    sentiments = [s for _, _, s in window if s is not None]
    mean_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

    burst = len(window) >= SHOCK_MIN_EVENTS and rate_ratio >= SHOCK_RATE_RATIO
    shock_type: Optional[str] = None
    if burst and mean_sentiment <= -SHOCK_SENTIMENT_ABS:
        shock_type = "panic"
    elif burst and mean_sentiment >= SHOCK_SENTIMENT_ABS:
        shock_type = "euphoria"

    # 0..1 intensity: how far past both thresholds, capped.
    intensity = 0.0
    if shock_type is not None:
        rate_part = min(1.0, rate_ratio / (SHOCK_RATE_RATIO * 2))
        sentiment_part = min(1.0, abs(mean_sentiment) / (SHOCK_SENTIMENT_ABS * 2))
        intensity = round(0.5 * rate_part + 0.5 * sentiment_part, 4)

    return {
        "shock": shock_type is not None,
        "type": shock_type,
        "intensity": intensity,
        "window_events": len(window),
        "baseline_events": len(baseline),
        "rate_ratio": round(rate_ratio, 2),
        "mean_sentiment": round(mean_sentiment, 4),
        "window_seconds": int(SHOCK_WINDOW_SECONDS),
    }

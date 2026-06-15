"""Multi-horizon sentiment outlook scoring.

Produces a bearish(-1)..bullish(+1) OUTLOOK per symbol over 1h/6h/24h
horizons, blended into a composite with a small market-context tilt. This is
an analytical opinion about mood and momentum of attention - it is NOT a
trade signal and must never be interpreted as "buy" or "sell".

Design rules:
- Pure functions over plain rows; fully unit-testable offline.
- Low data -> low confidence, score pulled toward 0. Never confident on thin data.
- Context (Fear & Greed, market cap trend) tilts by at most CONTEXT_MAX_TILT.
- Every component is reported so the dashboard can show WHY.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

from sentiment_engine.config import (
    CALIBRATION_MODEL_PATH,
    CONTEXT_MAX_TILT,
    NEWS_SOURCE_WEIGHT,
    OUTLOOK_HORIZON_WEIGHTS,
    OUTLOOK_HORIZONS_SECONDS_T,
)
from sentiment_engine.types import ContextSnapshot

# Cached calibration model, reloaded when the file's mtime changes (so a fresh
# `calibrate` run is picked up without restarting the engine).
_MODEL_CACHE: Dict[str, object] = {"mtime": None, "model": None}


def load_calibration_model(path: str = CALIBRATION_MODEL_PATH):
    """Return the adopted CalibrationModel, or None (default scorer) if absent."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        _MODEL_CACHE.update(mtime=None, model=None)
        return None
    if _MODEL_CACHE["mtime"] != mtime:
        from sentiment_engine.signals.calibration import CalibrationModel
        try:
            with open(path, encoding="utf-8") as fh:
                model = CalibrationModel.from_json(fh.read())
            _MODEL_CACHE.update(mtime=mtime, model=model if model.adopted else None)
        except Exception:  # noqa: BLE001 - a bad model must never break scoring
            _MODEL_CACHE.update(mtime=mtime, model=None)
    return _MODEL_CACHE["model"]

EventRow = Tuple[float, str, Optional[float]]  # (timestamp, source, sentiment)

# Confidence saturates at this many weighted events per horizon.
_FULL_CONFIDENCE_EVENTS = 30.0


def _weight(source: str) -> float:
    return NEWS_SOURCE_WEIGHT if source.startswith("news:") else 1.0


def weighted_average_sentiment(rows: Sequence[EventRow]) -> Optional[float]:
    scored = [(s, _weight(src)) for _, src, s in rows if s is not None]
    if not scored:
        return None
    total_weight = sum(w for _, w in scored)
    return sum(s * w for s, w in scored) / total_weight


def horizon_aggregate(
    rows: Sequence[EventRow], now: float, horizon_seconds: int, model: object = None
) -> Dict[str, object]:
    """Score one horizon from its event rows.

    Default: score = 0.7 * weighted avg sentiment + 0.3 * trend (second half vs
    first half of the window), clamped to [-1, 1]. If a calibration `model` with
    a learned head for this horizon is supplied, its fitted mapping replaces the
    0.7/0.3 coefficients. Either way the score is scaled by confidence, so thin
    data is pulled toward neutral. Confidence grows with weighted event count and
    source diversity.
    """
    start = now - horizon_seconds
    window = [row for row in rows if start < row[0] <= now]
    midpoint = now - horizon_seconds / 2
    first_half = [row for row in window if row[0] <= midpoint]
    second_half = [row for row in window if row[0] > midpoint]

    average = weighted_average_sentiment(window)
    first_avg = weighted_average_sentiment(first_half)
    second_avg = weighted_average_sentiment(second_half)
    trend = (second_avg - first_avg) if (first_avg is not None and second_avg is not None) else 0.0

    weighted_count = sum(_weight(src) for _, src, s in window if s is not None)
    distinct_sources = len({src for _, src, _ in window})
    coverage = min(1.0, weighted_count / _FULL_CONFIDENCE_EVENTS)
    diversity = 0.7 + 0.3 * min(1.0, max(0, distinct_sources - 1) / 2.0)
    confidence = round(coverage * diversity, 4)

    if average is None:
        score = 0.0
    else:
        head = getattr(model, "horizons", {}).get(horizon_seconds) if model is not None else None
        if head is not None:
            base = head.score(average, trend)  # learned, already clamped to [-1, 1]
        else:
            base = max(-1.0, min(1.0, 0.7 * average + 0.3 * trend))
        score = base * confidence  # thin data pulls the score toward neutral
    return {
        "horizon_seconds": horizon_seconds,
        "calibrated": bool(getattr(model, "horizons", {}).get(horizon_seconds)) if model is not None else False,
        "score": round(score, 4),
        "average_sentiment": round(average, 4) if average is not None else None,
        "trend": round(trend, 4),
        "event_count": len(window),
        "news_count": sum(1 for _, src, _ in window if src.startswith("news:")),
        "distinct_sources": distinct_sources,
        "confidence": confidence,
    }


def context_tilt(context: Optional[ContextSnapshot], stale: bool) -> float:
    """Small additive tilt from market context. Stale/missing context -> 0.

    Fear & Greed above 50 tilts bullish, below tilts bearish (mood-following,
    not contrarian - regime flips are the trend layer's job, not v1's).
    24h market cap change adds a momentum component.
    """
    if context is None or stale:
        return 0.0
    tilt = 0.0
    if context.fear_greed_value is not None:
        tilt += ((context.fear_greed_value - 50) / 50.0) * (CONTEXT_MAX_TILT * 0.6)
    if context.market_cap_change_24h_pct is not None:
        change = max(-5.0, min(5.0, context.market_cap_change_24h_pct))
        tilt += (change / 5.0) * (CONTEXT_MAX_TILT * 0.4)
    return round(max(-CONTEXT_MAX_TILT, min(CONTEXT_MAX_TILT, tilt)), 4)


def composite_outlook(
    horizons: List[Dict[str, object]],
    tilt: float,
    model: object = None,
    lsr_signal: Optional[float] = None,
    funding_signal: Optional[float] = None,
) -> Tuple[float, float]:
    """(composite score, composite confidence) across weighted horizons.

    A calibration `model` may supply learned per-horizon blend weights, a tilt
    coefficient, and a crowd-positioning coefficient (`lsr_coef`) applied to
    `lsr_signal`. The positioning term only contributes once the learner has
    adopted a non-zero coefficient, so its sign (momentum vs contrarian) is
    learned from outcomes, not assumed. Absent a model, behaviour is unchanged.
    """
    if model is not None and getattr(model, "horizon_weights", None):
        weights = [model.horizon_weights.get(int(h["horizon_seconds"]), 0.0) for h in horizons]
        tilt_coef = getattr(model, "tilt_coef", 1.0)
        lsr_coef = getattr(model, "lsr_coef", 0.0)
        funding_coef = getattr(model, "funding_coef", 0.0)
    else:
        weights = list(OUTLOOK_HORIZON_WEIGHTS[: len(horizons)])
        tilt_coef = 1.0
        lsr_coef = 0.0
        funding_coef = 0.0
    total = sum(weights) or 1.0
    confidence = sum(float(h["confidence"]) * w for h, w in zip(horizons, weights)) / total

    logistic = getattr(model, "logistic", None) if model is not None else None
    if logistic is not None:
        # Adopted multi-feature model supersedes the linear blend + coefficients.
        from sentiment_engine.signals.logistic_model import extract_logistic_features
        feats = extract_logistic_features(horizons, lsr_signal, funding_signal)
        score = logistic.score(feats)
    else:
        score = sum(float(h["score"]) * w for h, w in zip(horizons, weights)) / total
        score += tilt * tilt_coef
        if lsr_signal is not None and lsr_coef:
            score += float(lsr_signal) * lsr_coef
        if funding_signal is not None and funding_coef:
            score += float(funding_signal) * funding_coef
    score = max(-1.0, min(1.0, score))
    return round(score, 4), round(confidence, 4)


def outlook_label(score: float, confidence: float) -> str:
    if confidence < 0.15:
        return "insufficient data"
    if score >= 0.35:
        return "bullish"
    if score >= 0.12:
        return "mildly bullish"
    if score <= -0.35:
        return "bearish"
    if score <= -0.12:
        return "mildly bearish"
    return "neutral"


def horizons_config() -> Tuple[int, ...]:
    return OUTLOOK_HORIZONS_SECONDS_T

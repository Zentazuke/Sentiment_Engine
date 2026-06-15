"""Standalone FastAPI service for sentiment signal evaluation.

Run:
    uvicorn sentiment_engine.api:app --host 127.0.0.1 --port 8787
"""

from __future__ import annotations

import json as _json
import time
from decimal import Decimal
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sentiment_engine.aggregation.state_store import STATE
from sentiment_engine.config import SERVICE_PORT, SUPPORTED_SYMBOLS
from sentiment_engine.processing.coin_mapper import normalize_symbol
from sentiment_engine.processing.scoring import score_text
from sentiment_engine.signals.decision import evaluate_trade_signal
from sentiment_engine.storage.micro_history import get_micro_history
from sentiment_engine.storage.social_history import get_social_history
from sentiment_engine.storage.outcome_tracker import OutcomeTracker
from sentiment_engine.storage.sqlite_journal import JOURNAL
from sentiment_engine.types import ContextSnapshot, MicrostructureSnapshot, PriceEvent, SocialEvent, TradeDirection

OUTCOME_TRACKER = OutcomeTracker(JOURNAL.fill_outcome, STATE.latest_price)

app = FastAPI(
    title="Crypto Scalping Sentiment Engine",
    version="0.1.0",
    description="Standalone sentiment confirmation layer. It never places trades.",
)


class SocialIngestRequest(BaseModel):
    symbol: str
    source: str = "manual"
    text: str
    author: Optional[str] = None
    timestamp: Optional[float] = None
    # Optional explicit sentiment for sources that carry their own label
    # (e.g. CryptoPanic crowd votes). When omitted, the text is scored.
    sentiment: Optional[float] = Field(default=None, ge=-1, le=1)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class PriceIngestRequest(BaseModel):
    symbol: str
    price: float = Field(gt=0)
    timestamp: Optional[float] = None


class MicrostructureIngestRequest(BaseModel):
    """Microstructure metrics pushed by a live feed process.

    All metric fields are optional: absent/None means "unavailable" and the
    engine treats it as missing data, never as zero.
    """

    symbol: str
    computed_at: Optional[float] = None
    last_price: Optional[float] = Field(default=None, gt=0)
    buy_volume_10s: Optional[float] = Field(default=None, ge=0)
    sell_volume_10s: Optional[float] = Field(default=None, ge=0)
    trade_imbalance_10s: Optional[float] = Field(default=None, ge=-1, le=1)
    trade_imbalance_30s: Optional[float] = Field(default=None, ge=-1, le=1)
    trade_imbalance_60s: Optional[float] = Field(default=None, ge=-1, le=1)
    relative_volume: Optional[float] = Field(default=None, ge=0)
    momentum_pct_30s: Optional[float] = None
    volatility_bps_30s: Optional[float] = Field(default=None, ge=0)
    vwap_distance_bps: Optional[float] = None
    bid_depth_quote: Optional[float] = Field(default=None, ge=0)
    ask_depth_quote: Optional[float] = Field(default=None, ge=0)
    book_imbalance: Optional[float] = Field(default=None, ge=-1, le=1)
    spread_bps: Optional[float] = Field(default=None, ge=0)
    trade_count_60s: Optional[int] = Field(default=None, ge=0)


class EvaluateRequest(BaseModel):
    symbol: str
    direction: TradeDirection
    bot_confidence: float = Field(ge=0, le=1)
    trigger_price: Optional[float] = Field(default=None, gt=0)
    timestamp: Optional[float] = None


def _decimal(value: float | int | str) -> Decimal:
    return Decimal(str(value))


def _snapshot_to_dict(snapshot: Any) -> Dict[str, Any]:
    return {
        "symbol": snapshot.symbol,
        "window_seconds": snapshot.window_seconds,
        "message_count": snapshot.message_count,
        "previous_message_count": snapshot.previous_message_count,
        "average_sentiment": float(snapshot.average_sentiment),
        "previous_average_sentiment": float(snapshot.previous_average_sentiment),
        "sentiment_velocity": float(snapshot.sentiment_velocity),
        "attention_spike": float(snapshot.attention_spike),
        "price_change_pct": float(snapshot.price_change_pct),
        "data_quality": float(snapshot.data_quality),
    }




_MICRO_METRIC_FIELDS = (
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


def _microstructure_dict(symbol: str) -> Optional[Dict[str, Any]]:
    """Latest microstructure block for a symbol, or None if never ingested."""
    micro, stale = STATE.microstructure(symbol)
    if micro is None:
        return None
    payload: Dict[str, Any] = {field: getattr(micro, field) for field in _MICRO_METRIC_FIELDS}
    payload["computed_at"] = micro.computed_at
    payload["stale"] = stale
    return payload


@app.get("/health")
def health() -> Dict[str, Any]:
    now = time.time()
    feeds: Dict[str, Any] = {}
    for symbol in SUPPORTED_SYMBOLS:
        micro, stale = STATE.microstructure(symbol)
        feeds[symbol.replace("/", "-")] = {
            "microstructure_age_seconds": round(now - micro.computed_at, 1) if micro else None,
            "stale": stale,
            "live": micro is not None and not stale,
        }
    return {
        "status": "ok",
        "mode": "standalone",
        "symbols": list(SUPPORTED_SYMBOLS),
        "port": SERVICE_PORT,
        "safety": "does_not_trade",
        "feeds": feeds,
    }


@app.post("/ingest/social")
def ingest_social(payload: SocialIngestRequest) -> Dict[str, Any]:
    try:
        symbol = normalize_symbol(payload.symbol)
        if payload.sentiment is not None:
            sentiment = _decimal(payload.sentiment)
            confidence = _decimal(payload.confidence if payload.confidence is not None else 0.8)
        else:
            sentiment, confidence = score_text(payload.text)
        event = SocialEvent(
            symbol=symbol,
            source=payload.source,
            text=payload.text,
            author=payload.author,
            timestamp=payload.timestamp if payload.timestamp is not None else time.time(),
            sentiment=sentiment,
            confidence=confidence,
        )
        STATE.add_social(event)
        get_social_history().record_event(event)
        return {
            "accepted": True,
            "symbol": symbol,
            "sentiment": float(sentiment),
            "confidence": float(confidence),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest/price")
def ingest_price(payload: PriceIngestRequest) -> Dict[str, Any]:
    try:
        symbol = normalize_symbol(payload.symbol)
        event = PriceEvent(
            symbol=symbol,
            price=_decimal(payload.price),
            timestamp=payload.timestamp if payload.timestamp is not None else time.time(),
        )
        STATE.add_price(event)
        return {"accepted": True, "symbol": symbol}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest/microstructure")
def ingest_microstructure(payload: MicrostructureIngestRequest) -> Dict[str, Any]:
    try:
        symbol = normalize_symbol(payload.symbol)
        now = time.time()
        snapshot = MicrostructureSnapshot(
            symbol=symbol,
            computed_at=payload.computed_at if payload.computed_at is not None else now,
            **{field: getattr(payload, field) for field in _MICRO_METRIC_FIELDS},
        )
        STATE.add_microstructure(snapshot, received_at=now)
        get_micro_history().record(snapshot, received_at=now)
        return {"accepted": True, "symbol": symbol}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/snapshot/{symbol_key}")
def snapshot(symbol_key: str) -> Dict[str, Any]:
    try:
        snap = STATE.snapshot(symbol_key)
        result = _snapshot_to_dict(snap)
        result["microstructure"] = _microstructure_dict(symbol_key)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/social/{symbol_key}/recent")
def recent_social(symbol_key: str, limit: int = 20) -> Dict[str, Any]:
    """Most recent social/news events for one symbol (display only)."""
    try:
        symbol = normalize_symbol(symbol_key)
        capped = max(1, min(50, limit))
        events = STATE.recent_social(symbol, limit=capped)
        payload = [
            {
                "source": event.source,
                "text": event.text,
                "author": event.author,
                "timestamp": event.timestamp,
                "sentiment": float(event.sentiment) if event.sentiment is not None else None,
            }
            for event in events
        ]
        if not payload:
            # Memory buffer empties on restart; fall back to persistent history.
            payload = [
                {"source": src, "text": text, "author": author, "timestamp": ts, "sentiment": sentiment}
                for ts, src, author, text, sentiment in get_social_history().recent_events(symbol, capped)
            ]
        return {"symbol": symbol, "events": payload}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class ContextIngestRequest(BaseModel):
    fear_greed_value: Optional[int] = Field(default=None, ge=0, le=100)
    fear_greed_label: Optional[str] = None
    btc_dominance_pct: Optional[float] = Field(default=None, ge=0, le=100)
    total_market_cap_usd: Optional[float] = Field(default=None, ge=0)
    market_cap_change_24h_pct: Optional[float] = None
    fetched_at: Optional[float] = None


def _context_dict() -> Optional[Dict[str, Any]]:
    context, stale = STATE.context()
    if context is None:
        return None
    return {
        "fear_greed_value": context.fear_greed_value,
        "fear_greed_label": context.fear_greed_label,
        "btc_dominance_pct": context.btc_dominance_pct,
        "total_market_cap_usd": context.total_market_cap_usd,
        "market_cap_change_24h_pct": context.market_cap_change_24h_pct,
        "fetched_at": context.fetched_at,
        "stale": stale,
    }


@app.post("/ingest/context")
def ingest_context(payload: ContextIngestRequest) -> Dict[str, Any]:
    now = time.time()
    snapshot = ContextSnapshot(
        fetched_at=payload.fetched_at if payload.fetched_at is not None else now,
        fear_greed_value=payload.fear_greed_value,
        fear_greed_label=payload.fear_greed_label,
        btc_dominance_pct=payload.btc_dominance_pct,
        total_market_cap_usd=payload.total_market_cap_usd,
        market_cap_change_24h_pct=payload.market_cap_change_24h_pct,
    )
    STATE.add_context(snapshot, received_at=now)
    return {"accepted": True}


@app.get("/context")
def context() -> Dict[str, Any]:
    return {"context": _context_dict()}


class PositioningIngestRequest(BaseModel):
    symbol: str
    global_account_ratio: Optional[float] = Field(default=None, ge=0)
    top_account_ratio: Optional[float] = Field(default=None, ge=0)
    top_position_ratio: Optional[float] = Field(default=None, ge=0)
    taker_ratio: Optional[float] = Field(default=None, ge=0)
    timestamp: Optional[float] = None


@app.post("/ingest/positioning")
def ingest_positioning(payload: PositioningIngestRequest) -> Dict[str, Any]:
    from sentiment_engine.storage.positioning_history import get_positioning_history
    try:
        symbol = normalize_symbol(payload.symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    get_positioning_history().record(
        symbol=symbol,
        global_account_ratio=payload.global_account_ratio,
        top_account_ratio=payload.top_account_ratio,
        top_position_ratio=payload.top_position_ratio,
        taker_ratio=payload.taker_ratio,
        timestamp=payload.timestamp,
    )
    return {"accepted": True}


@app.get("/positioning/{symbol_key}")
def positioning(symbol_key: str) -> Dict[str, Any]:
    """Latest crowd long/short positioning for a symbol. Descriptive only."""
    from sentiment_engine.storage.positioning_history import get_positioning_history
    try:
        symbol = normalize_symbol(symbol_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"symbol": symbol, "positioning": get_positioning_history().latest(symbol)}


class DerivativesIngestRequest(BaseModel):
    symbol: str
    funding_rate: Optional[float] = None
    open_interest_usd: Optional[float] = Field(default=None, ge=0)
    timestamp: Optional[float] = None


@app.post("/ingest/derivatives")
def ingest_derivatives(payload: DerivativesIngestRequest) -> Dict[str, Any]:
    from sentiment_engine.storage.derivatives_history import get_derivatives_history
    try:
        symbol = normalize_symbol(payload.symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    get_derivatives_history().record(
        symbol=symbol,
        funding_rate=payload.funding_rate,
        open_interest_usd=payload.open_interest_usd,
        timestamp=payload.timestamp,
    )
    return {"accepted": True}


@app.get("/derivatives/{symbol_key}")
def derivatives(symbol_key: str) -> Dict[str, Any]:
    """Latest funding rate + open interest for a symbol. Descriptive only."""
    from sentiment_engine.storage.derivatives_history import get_derivatives_history
    try:
        symbol = normalize_symbol(symbol_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"symbol": symbol, "derivatives": get_derivatives_history().latest(symbol)}


@app.get("/outlook/{symbol_key}/history")
def outlook_history(symbol_key: str, hours: float = 12.0) -> Dict[str, Any]:
    """Journaled outlook scores + prices for the timeline chart."""
    try:
        symbol = normalize_symbol(symbol_key)
        since = time.time() - max(0.5, min(168.0, hours)) * 3600.0
        rows = get_social_history().outlook_rows_since(symbol, since)
        return {
            "symbol": symbol,
            "points": [
                {"t": computed_at, "score": score, "confidence": confidence, "price": price}
                for computed_at, score, confidence, price in rows
            ],
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/insights/{symbol_key}")
def insights(symbol_key: str, hours: float = 24.0) -> Dict[str, Any]:
    """Trending terms + news/reddit sentiment split. Descriptive only."""
    from sentiment_engine.signals.insights import source_breakdown, trending_terms

    try:
        symbol = normalize_symbol(symbol_key)
        now = time.time()
        span = max(0.5, min(168.0, hours)) * 3600.0
        rows = get_social_history().text_rows_between(symbol, now - span, now)
        return {
            "symbol": symbol,
            "hours": span / 3600.0,
            "event_count": len(rows),
            "terms": trending_terms(rows),
            "sources": source_breakdown(rows),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/outlook/{symbol_key}")
def outlook(symbol_key: str) -> Dict[str, Any]:
    """Multi-horizon sentiment outlook. Analytical only - never a trade signal."""
    from sentiment_engine.signals.outlook import (
        composite_outlook,
        context_tilt,
        horizon_aggregate,
        horizons_config,
        load_calibration_model,
        outlook_label,
    )

    try:
        symbol = normalize_symbol(symbol_key)
        now = time.time()
        longest = max(horizons_config())
        rows = get_social_history().events_between(symbol, now - longest, now, exclude_synthetic=True)
        model = load_calibration_model()
        horizons = [horizon_aggregate(rows, now, h, model) for h in horizons_config()]
        ctx, ctx_stale = STATE.context()
        tilt = context_tilt(ctx, ctx_stale)
        from sentiment_engine.storage.positioning_history import get_positioning_history
        from sentiment_engine.storage.derivatives_history import get_derivatives_history
        pos = get_positioning_history().latest(symbol)
        lsr_signal = pos.get("signal") if pos else None
        deriv = get_derivatives_history().latest(symbol)
        funding_signal = deriv.get("funding_signal") if deriv else None
        score, confidence = composite_outlook(horizons, tilt, model, lsr_signal, funding_signal)
        direction_prob_6h = None
        if model is not None and getattr(model, "logistic", None) is not None:
            from sentiment_engine.signals.logistic_model import extract_logistic_features
            feats = extract_logistic_features(horizons, lsr_signal, funding_signal)
            direction_prob_6h = round(model.logistic.probability(feats), 4)
        result = {
            "symbol": symbol,
            "computed_at": now,
            "outlook_score": score,
            "confidence": confidence,
            "label": outlook_label(score, confidence),
            "context_tilt": tilt,
            "horizons": horizons,
            "context": _context_dict(),
            "positioning": pos,
            "direction_prob_6h": direction_prob_6h,
            "calibrated": model is not None,
            "disclaimer": "Analytical outlook only. Not a trade signal.",
        }
        get_social_history().record_outlook(
            symbol=symbol,
            computed_at=now,
            composite_score=score,
            confidence=confidence,
            context_tilt=tilt,
            horizons_json=_json.dumps(horizons),
            price_at_compute=STATE.latest_price(symbol),
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/alerts/{symbol_key}")
def alerts(symbol_key: str) -> Dict[str, Any]:
    """Shock/panic detection over recent social activity. Descriptive only."""
    from sentiment_engine.config import SHOCK_BASELINE_SECONDS, SHOCK_WINDOW_SECONDS
    from sentiment_engine.signals.shock import detect_shock

    try:
        symbol = normalize_symbol(symbol_key)
        now = time.time()
        span = SHOCK_WINDOW_SECONDS + SHOCK_BASELINE_SECONDS
        rows = get_social_history().events_between(symbol, now - span, now, exclude_synthetic=True)
        result = detect_shock(rows, now)
        result["symbol"] = symbol
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/evaluate")
def evaluate(payload: EvaluateRequest) -> Dict[str, Any]:
    """Evaluate an existing bot signal.

    Safe failure rule: any internal exception returns neutral-style values by
    design. Unsupported symbols/directions still raise 400 because that is a
    caller contract error.
    """
    try:
        symbol = normalize_symbol(payload.symbol)
        if payload.trigger_price is not None:
            STATE.add_price(
                PriceEvent(
                    symbol=symbol,
                    price=_decimal(payload.trigger_price),
                    timestamp=payload.timestamp if payload.timestamp is not None else time.time(),
                )
            )
        snap = STATE.snapshot(symbol, now=payload.timestamp)
        micro, micro_stale = STATE.microstructure(symbol, now=payload.timestamp)
        bot_conf = _decimal(payload.bot_confidence)
        result = evaluate_trade_signal(
            snapshot=snap,
            direction=payload.direction,
            bot_confidence=bot_conf,
            micro=micro,
            micro_stale=micro_stale,
        )
        eval_time = payload.timestamp if payload.timestamp is not None else time.time()
        row_id = JOURNAL.log_evaluation(
            evaluation=result,
            direction=payload.direction,
            bot_confidence=bot_conf,
            timestamp=payload.timestamp,
            price_at_eval=STATE.latest_price(symbol),
        )
        if row_id:
            OUTCOME_TRACKER.register(row_id, symbol, eval_time)
        return {
            "symbol": result.symbol,
            "action": result.action.value,
            "sentiment_score": float(result.sentiment_score),
            "confidence_modifier": float(result.confidence_modifier),
            "final_confidence_estimate": float(result.final_confidence_estimate),
            "data_quality": float(result.data_quality),
            "reason": result.reason,
            "safe_to_use": result.safe_to_use,
            "snapshot": _snapshot_to_dict(result.snapshot),
            "microstructure": _microstructure_dict(symbol),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - safe failure boundary
        # Do not pretend this is usable, and never veto/confirm on failure.
        return {
            "symbol": payload.symbol,
            "action": "neutral",
            "sentiment_score": 0.0,
            "confidence_modifier": 1.0,
            "final_confidence_estimate": payload.bot_confidence,
            "data_quality": 0.0,
            "reason": f"Neutral: sentiment engine error; defaulted safe. {type(exc).__name__}",
            "safe_to_use": False,
            "snapshot": None,
        }

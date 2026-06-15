"""Tests for outlook scoring, social history, context, and connectors."""

import time

import pytest

from sentiment_engine.signals.outlook import (
    composite_outlook,
    context_tilt,
    horizon_aggregate,
    outlook_label,
    weighted_average_sentiment,
)
from sentiment_engine.types import ContextSnapshot

NOW = 1_000_000.0


def rows_at(offsets_sentiments, source="dashboard"):
    return [(NOW - offset, source, sentiment) for offset, sentiment in offsets_sentiments]


# --- horizon scoring -------------------------------------------------------

def test_empty_window_is_neutral_zero_confidence():
    agg = horizon_aggregate([], NOW, 3600)
    assert agg["score"] == 0.0
    assert agg["confidence"] == 0.0


def test_bullish_events_score_positive():
    rows = rows_at([(i * 60, 0.6) for i in range(40)])
    agg = horizon_aggregate(rows, NOW, 3600)
    assert agg["score"] > 0.2
    assert agg["confidence"] > 0.5


def test_thin_data_pulls_score_toward_zero():
    few = horizon_aggregate(rows_at([(60, 0.8), (120, 0.8)]), NOW, 3600)
    many = horizon_aggregate(rows_at([(i * 60, 0.8) for i in range(40)]), NOW, 3600)
    assert abs(few["score"]) < abs(many["score"])


def test_news_weighted_heavier():
    social = [(NOW - 60, "dashboard", 0.5), (NOW - 60, "dashboard", 0.5)]
    mixed = social + [(NOW - 60, "news:coindesk.com", -0.5)]
    avg = weighted_average_sentiment(mixed)
    assert avg == pytest.approx((0.5 + 0.5 - 0.5 * 2.0) / 4.0)  # news weight 2.0


def test_trend_detected():
    rows = rows_at([(3000, -0.4)] * 10 + [(300, 0.5)] * 10)
    agg = horizon_aggregate(rows, NOW, 3600)
    assert agg["trend"] > 0.5


# --- context tilt -----------------------------------------------------------

def make_context(**kwargs):
    defaults = dict(fetched_at=NOW)
    defaults.update(kwargs)
    return ContextSnapshot(**defaults)


def test_stale_or_missing_context_no_tilt():
    assert context_tilt(None, stale=True) == 0.0
    assert context_tilt(make_context(fear_greed_value=90), stale=True) == 0.0


def test_greed_tilts_bullish_fear_bearish():
    assert context_tilt(make_context(fear_greed_value=90), stale=False) > 0
    assert context_tilt(make_context(fear_greed_value=10), stale=False) < 0


def test_tilt_bounded():
    extreme = make_context(fear_greed_value=100, market_cap_change_24h_pct=50.0)
    assert context_tilt(extreme, stale=False) <= 0.10


def test_composite_blends_and_clamps():
    horizons = [
        {"score": 0.8, "confidence": 0.9},
        {"score": 0.4, "confidence": 0.5},
        {"score": 0.2, "confidence": 0.3},
    ]
    score, confidence = composite_outlook(horizons, tilt=0.05)
    assert 0.4 < score < 0.7
    assert 0.4 < confidence < 0.8


def test_labels():
    assert outlook_label(0.5, 0.8) == "bullish"
    assert outlook_label(-0.5, 0.8) == "bearish"
    assert outlook_label(0.0, 0.8) == "neutral"
    assert outlook_label(0.9, 0.05) == "insufficient data"


# --- social history persistence ---------------------------------------------

def test_social_history_roundtrip(tmp_path):
    from decimal import Decimal
    from sentiment_engine.storage.social_history import SocialHistory
    from sentiment_engine.types import SocialEvent

    history = SocialHistory(tmp_path / "hist.db")
    history.record_event(SocialEvent(
        symbol="BTC/USDT", source="news:test.com", text="Bitcoin rallies",
        timestamp=NOW - 100, sentiment=Decimal("0.5"), confidence=Decimal("0.8"),
    ))
    rows = history.events_between("BTC/USDT", NOW - 3600, NOW)
    history.close()
    assert rows == [(NOW - 100, "news:test.com", 0.5)]


# --- connector parsers --------------------------------------------------------

def test_parse_fng():
    from sentiment_engine.ingestion.context_feed import parse_fng

    data = {"data": [{"value": "72", "value_classification": "Greed"}]}
    assert parse_fng(data) == {"fear_greed_value": 72, "fear_greed_label": "Greed"}
    assert parse_fng({}) == {}


def test_parse_coingecko_global():
    from sentiment_engine.ingestion.context_feed import parse_coingecko_global

    data = {"data": {
        "market_cap_percentage": {"btc": 54.3},
        "total_market_cap": {"usd": 2.5e12},
        "market_cap_change_percentage_24h_usd": -1.2,
    }}
    parsed = parse_coingecko_global(data)
    assert parsed["btc_dominance_pct"] == pytest.approx(54.3)
    assert parsed["market_cap_change_24h_pct"] == pytest.approx(-1.2)


def test_reddit_listing_prime_then_new():
    from sentiment_engine.ingestion.reddit_feed import RedditFeed

    feed = RedditFeed(["BTC/USDT", "ADA/USDT"], engine_url="http://test")
    listing = {"data": {"children": [
        {"data": {"name": "t3_1", "title": "Bitcoin to the moon", "created_utc": NOW}},
        {"data": {"name": "t3_2", "title": "Best altcoins?", "created_utc": NOW}},
    ]}}
    assert feed.process_listing("Bitcoin", listing) == []  # primed silently
    listing["data"]["children"].append(
        {"data": {"name": "t3_3", "title": "Cardano staking question", "created_utc": NOW}})
    assert feed.process_listing("Bitcoin", listing) == [("ADA/USDT", "Cardano staking question")]


# --- API endpoints --------------------------------------------------------------

def test_context_and_outlook_endpoints():
    from fastapi.testclient import TestClient
    from sentiment_engine.api import app

    client = TestClient(app)
    response = client.post("/ingest/context", json={
        "fear_greed_value": 75, "fear_greed_label": "Greed",
        "btc_dominance_pct": 55.0, "market_cap_change_24h_pct": 2.0,
    })
    assert response.status_code == 200
    ctx = client.get("/context").json()["context"]
    assert ctx["fear_greed_value"] == 75
    assert ctx["stale"] is False

    for i in range(25):
        client.post("/ingest/social", json={
            "symbol": "BTC/USDT", "source": "news:test.com",
            "text": "Bitcoin breakout looks strong, massive pump",
            "timestamp": time.time() - i * 60,
        })
    data = client.get("/outlook/BTC-USDT").json()
    assert data["outlook_score"] > 0
    assert data["label"] in ("bullish", "mildly bullish", "neutral")
    assert len(data["horizons"]) == 3
    assert data["context_tilt"] > 0
    assert "Not a trade signal" in data["disclaimer"]

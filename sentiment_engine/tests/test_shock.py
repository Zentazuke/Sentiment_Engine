"""Tests for shock detection."""

import time

from sentiment_engine.signals.shock import detect_shock

NOW = 1_000_000.0


def burst(count, sentiment, offset_max=500):
    return [(NOW - (i % offset_max) - 1, "reddit:Bitcoin", sentiment) for i in range(count)]


def baseline(count, sentiment=0.0):
    # Spread across the baseline hour before the window.
    return [(NOW - 600 - 1 - i * (3600 / max(count, 1)), "reddit:Bitcoin", sentiment) for i in range(count)]


def test_quiet_market_no_shock():
    result = detect_shock(baseline(10), NOW)
    assert result["shock"] is False
    assert result["type"] is None


def test_panic_burst_detected():
    rows = baseline(10) + burst(20, -0.6)
    result = detect_shock(rows, NOW)
    assert result["shock"] is True
    assert result["type"] == "panic"
    assert result["intensity"] > 0.3


def test_euphoria_burst_detected():
    rows = baseline(10) + burst(20, 0.6)
    result = detect_shock(rows, NOW)
    assert result["type"] == "euphoria"


def test_burst_without_extreme_sentiment_is_not_shock():
    rows = baseline(10) + burst(20, 0.05)
    result = detect_shock(rows, NOW)
    assert result["shock"] is False


def test_extreme_sentiment_without_burst_is_not_shock():
    # High volume in window but similar baseline rate -> no burst.
    rows = baseline(200, -0.6) + burst(20, -0.6)
    result = detect_shock(rows, NOW)
    assert result["shock"] is False


def test_min_events_required():
    result = detect_shock(burst(4, -0.9), NOW)
    assert result["shock"] is False


def test_alerts_endpoint():
    from fastapi.testclient import TestClient
    from sentiment_engine.api import app

    client = TestClient(app)
    now = time.time()
    for i in range(15):
        client.post("/ingest/social", json={
            "symbol": "ADA/USDT", "source": "reddit:cardano",
            "text": "ADA crashing hard, panic selloff, support lost",
            "timestamp": now - i * 20,
        })
    data = client.get("/alerts/ADA-USDT").json()
    assert data["symbol"] == "ADA/USDT"
    assert data["shock"] is True
    assert data["type"] == "panic"

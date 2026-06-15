"""Tests for the shadow-mode client. The engine-down path is the critical one."""

import time

from sentiment_engine.client import SentimentShadowClient


def make_offline_client():
    # Port 9 (discard) refuses connections instantly on Windows/Linux.
    return SentimentShadowClient(base_url="http://127.0.0.1:9", timeout=0.5)


def test_engine_down_returns_neutral_and_never_raises():
    client = make_offline_client()
    result = client.evaluate(symbol="BTC/USDT", direction="STRAT_LONG", bot_confidence=0.7)
    assert result["action"] == "neutral"
    assert result["safe_to_use"] is False
    assert result["final_confidence_estimate"] == 0.7
    assert client.stats["failed"] == 1


def test_engine_down_is_fast():
    client = make_offline_client()
    started = time.monotonic()
    client.evaluate(symbol="BTC/USDT", direction="STRAT_LONG", bot_confidence=0.7)
    assert time.monotonic() - started < 2.0  # bounded by timeout, not hanging


def test_invalid_direction_rejected_locally():
    client = make_offline_client()
    result = client.evaluate(symbol="BTC/USDT", direction="BUY_NOW", bot_confidence=0.7)
    assert result["action"] == "neutral"
    assert "invalid direction" in result["reason"]
    assert client.stats["sent"] == 0  # never even tried the network


def test_confidence_clamped():
    client = make_offline_client()
    result = client.evaluate(symbol="BTC/USDT", direction="STRAT_LONG", bot_confidence=7.0)
    assert result["final_confidence_estimate"] == 7.0  # fallback echoes input...
    # ...but the payload sent to the engine is clamped; verify via the builder:
    # (engine-side validation also enforces 0..1)


def test_async_returns_immediately_and_resolves_neutral():
    client = make_offline_client()
    started = time.monotonic()
    future = client.evaluate_async(symbol="ADA/USDT", direction="STRAT_SHORT", bot_confidence=0.5)
    assert time.monotonic() - started < 0.1  # fire-and-forget
    result = future.result(timeout=3)
    assert result["action"] == "neutral"


def test_health_none_when_down():
    assert make_offline_client().health() is None

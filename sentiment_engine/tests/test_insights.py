"""Tests for trending terms, source breakdown, and the insights endpoints."""

import time

from sentiment_engine.signals.insights import source_breakdown, trending_terms


def row(text, sentiment=0.5, source="news:x.com", ts=1000.0):
    return (ts, source, text, sentiment)


def test_trending_terms_counts_and_sentiment():
    rows = [
        row("ETF approval rally", 0.6),
        row("ETF inflows continue", 0.4),
        row("hack drains exchange", -0.8),
        row("exchange hack confirmed", -0.7),
        row("totally unrelated weather", None),
    ]
    terms = {t["term"]: t for t in trending_terms(rows)}
    assert terms["etf"]["count"] == 2
    assert terms["etf"]["avg_sentiment"] == 0.5
    assert terms["hack"]["count"] == 2
    assert terms["hack"]["avg_sentiment"] == -0.75
    assert "weather" not in terms  # count==1 filtered as noise


def test_trending_terms_ignores_stopwords_and_coin_names():
    rows = [row("the bitcoin price and the market", 0.1)] * 3
    terms = [t["term"] for t in trending_terms(rows)]
    assert "bitcoin" not in terms and "the" not in terms and "price" not in terms


def test_term_counted_once_per_message():
    rows = [row("pump pump pump", 0.5), row("pump again", 0.5)]
    terms = {t["term"]: t for t in trending_terms(rows)}
    assert terms["pump"]["count"] == 2  # messages, not occurrences


def test_source_breakdown_split_and_divergence():
    rows = [
        row("a", 0.6, "news:coindesk.com"),
        row("b", 0.4, "news:decrypt.co"),
        row("c", -0.5, "reddit:Bitcoin"),
        row("d", None, "reddit:Bitcoin"),
        row("e", 0.0, "telegram"),
    ]
    result = source_breakdown(rows)
    assert result["news"]["count"] == 2
    assert result["news"]["avg_sentiment"] == 0.5
    assert result["reddit"]["count"] == 2
    assert result["reddit"]["avg_sentiment"] == -0.5
    assert result["other"]["count"] == 1
    assert result["divergence"] == 1.0


def test_source_breakdown_handles_missing_classes():
    result = source_breakdown([row("a", 0.5, "news:x.com")])
    assert result["reddit"]["count"] == 0
    assert result["divergence"] is None


def test_insights_and_history_endpoints():
    from fastapi.testclient import TestClient
    from sentiment_engine.api import app

    client = TestClient(app)
    now = time.time()
    for i in range(3):
        client.post("/ingest/social", json={
            "symbol": "BTC/USDT", "source": "news:test.com",
            "text": "ETF approval pump rally", "timestamp": now - i * 60,
        })
    data = client.get("/insights/BTC-USDT?hours=1").json()
    assert data["event_count"] >= 3
    assert any(t["term"] == "etf" for t in data["terms"])
    assert data["sources"]["news"]["count"] >= 3

    client.get("/outlook/BTC-USDT")  # journals one outlook point
    history = client.get("/outlook/BTC-USDT/history?hours=1").json()
    assert len(history["points"]) >= 1
    assert "score" in history["points"][0]

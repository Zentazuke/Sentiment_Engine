"""Tests for the X connector: gating, budget, parsing, symbol matching."""

import pytest

from sentiment_engine.ingestion.x_feed import XFeed, build_query, match_symbols, parse_created_at


def make_feed(**kwargs):
    defaults = dict(symbols=["BTC/USDT", "ADA/USDT"], engine_url="http://test",
                    bearer_token="test-token", daily_read_budget=50)
    defaults.update(kwargs)
    return XFeed(**defaults)


def test_refuses_to_start_without_token():
    with pytest.raises(ValueError, match="SENTIMENT_X_BEARER_TOKEN"):
        XFeed(["BTC/USDT"], engine_url="http://test", bearer_token="")


def test_query_includes_cashtags_excludes_retweets():
    query = build_query(["BTC/USDT", "ADA/USDT"])
    assert "$BTC" in query and "$ADA" in query
    assert "-is:retweet" in query


def test_symbol_matching_cashtags_and_names():
    symbols = ["BTC/USDT", "ADA/USDT"]
    assert match_symbols("$BTC to 100k", symbols) == ["BTC/USDT"]
    assert match_symbols("Cardano season, $ada pumping", symbols) == ["ADA/USDT"]
    assert match_symbols("$BTC and $ADA both moving", symbols) == ["BTC/USDT", "ADA/USDT"]
    assert match_symbols("nothing relevant", symbols) == []


def test_process_response_parses_and_tracks_since_id():
    feed = make_feed()
    payload = {
        "data": [
            {"id": "2", "text": "$BTC breaking out hard", "created_at": "2026-06-12T10:00:00Z"},
            {"id": "1", "text": "$ADA looks weak, dumping", "created_at": "2026-06-12T09:59:00Z"},
            {"id": "0", "text": "unrelated stocks chat", "created_at": "2026-06-12T09:58:00Z"},
        ],
        "meta": {"newest_id": "2"},
    }
    pushes = feed.process_response(payload)
    assert [(s, t[:4]) for s, t, _ in pushes] == [("BTC/USDT", "$BTC"), ("ADA/USDT", "$ADA")]
    assert feed._since_id == "2"
    assert feed._reads_today == 3  # all returned tweets count against budget


def test_budget_enforced():
    feed = make_feed(daily_read_budget=5)
    feed.process_response({"data": [{"id": str(i), "text": "$BTC"} for i in range(5)],
                           "meta": {"newest_id": "4"}})
    assert feed._budget_remaining() == 0


def test_empty_response_costs_nothing():
    feed = make_feed()
    assert feed.process_response({"meta": {"result_count": 0}}) == []
    assert feed._reads_today == 0


def test_created_at_parsing():
    assert parse_created_at("2026-06-12T10:00:00.000Z") > 0
    assert parse_created_at(None) > 0  # falls back to now, never crashes
    assert parse_created_at("garbage") > 0

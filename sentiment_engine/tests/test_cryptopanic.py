"""CryptoPanic feed: vote->sentiment, parsing, priming, and the API override."""

from sentiment_engine.ingestion.cryptopanic_feed import (
    CryptoPanicFeed,
    parse_posts,
    supported_codes,
    vote_sentiment,
)


def test_vote_sentiment():
    assert vote_sentiment({"positive": 8, "negative": 2}) == 0.6
    assert vote_sentiment({"positive": 0, "negative": 4}) == -1.0
    assert vote_sentiment({"positive": 0, "negative": 0}) is None
    assert vote_sentiment(None) is None


def test_supported_codes_maps_bare_to_symbol():
    codes = supported_codes()
    assert codes["BTC"] == "BTC/USDT"
    assert codes["ETH"] == "ETH/USDT"


def test_parse_posts_extracts_symbols_and_votes():
    code_map = {"BTC": "BTC/USDT", "ETH": "ETH/USDT"}
    data = {"results": [
        {"id": 1, "title": "BTC ETF inflows surge", "votes": {"positive": 9, "negative": 1},
         "currencies": [{"code": "BTC"}]},
        {"id": 2, "title": "ETH and BTC update", "votes": {},
         "currencies": [{"code": "ETH"}, {"code": "BTC"}]},
        {"id": 3, "title": "DOGE moon", "currencies": [{"code": "DOGE"}]},  # unsupported -> dropped
    ]}
    parsed = parse_posts(data, code_map)
    keyed = {(pid, item[0]): item for pid, item in parsed}
    assert keyed[("1", "BTC/USDT")][2] == 0.8       # vote sentiment attached
    assert keyed[("2", "ETH/USDT")][2] is None      # no votes -> None (scored later)
    assert ("2", "BTC/USDT") in keyed
    assert all(it[0] != "DOGE" for _, it in parsed)


def _feed():
    return CryptoPanicFeed(["BTC/USDT", "ETH/USDT"], engine_url="http://x", token="t")


def test_enabled_requires_token():
    assert _feed().enabled is True
    assert CryptoPanicFeed(["BTC/USDT"], engine_url="http://x", token="").enabled is False


def test_process_primes_then_pushes_new():
    f = _feed()
    data = {"results": [{"id": 1, "title": "BTC pump", "votes": {"positive": 5, "negative": 0},
                         "currencies": [{"code": "BTC"}]}]}
    assert f.process(data) == []  # first poll primes, no push
    data2 = {"results": [{"id": 2, "title": "BTC rally", "votes": {"positive": 3, "negative": 1},
                          "currencies": [{"code": "BTC"}]}]}
    out = f.process(data2)
    assert out == [("BTC/USDT", "BTC rally", 0.5)]


def test_api_social_sentiment_override():
    from fastapi.testclient import TestClient
    from sentiment_engine.api import app
    client = TestClient(app)
    r = client.post("/ingest/social", json={
        "symbol": "BTC/USDT", "source": "cryptopanic", "text": "anything", "sentiment": -0.5,
    })
    assert r.status_code == 200
    assert abs(r.json()["sentiment"] + 0.5) < 1e-9  # used the override, not the scorer

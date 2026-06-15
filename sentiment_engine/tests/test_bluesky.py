"""Bluesky feed: Jetstream event parsing + strict coin matching."""

from sentiment_engine.ingestion.bluesky_feed import BlueskyFeed, extract_post_text


def _post_event(text):
    return {
        "kind": "commit",
        "commit": {"operation": "create", "collection": "app.bsky.feed.post", "record": {"text": text}},
    }


def _feed():
    return BlueskyFeed(["BTC/USDT", "ADA/USDT", "ETH/USDT", "BNB/USDT"], engine_url="http://x")


def test_extract_post_text():
    assert extract_post_text(_post_event("hello $BTC")) == "hello $BTC"


def test_extract_ignores_non_post_events():
    assert extract_post_text({"kind": "identity"}) is None
    assert extract_post_text({"kind": "commit", "commit": {"operation": "delete", "collection": "app.bsky.feed.post"}}) is None
    assert extract_post_text({"kind": "commit", "commit": {"operation": "create", "collection": "app.bsky.feed.like", "record": {}}}) is None


def test_matches_cashtag_and_name():
    f = _feed()
    assert f.match_symbols("loading up on $BTC today") == ["BTC/USDT"]
    assert "ETH/USDT" in f.match_symbols("ethereum upgrade looks bullish")
    assert f.match_symbols("cardano staking rewards rising") == ["ADA/USDT"]


def test_strict_matching_rejects_bare_tickers():
    f = _feed()
    # bare "ada"/"bnb"/"eth" must NOT match on the general firehose
    assert f.match_symbols("ada lovelace was a pioneer") == []
    assert f.match_symbols("the bnb tree grows tall") == []
    assert f.match_symbols("beth went to the eth zurich campus") == []


def test_process_event_returns_symbol_text_pairs():
    f = _feed()
    pairs = f.process_event(_post_event("$ETH and bitcoin both pumping"))
    syms = {s for s, _ in pairs}
    assert syms == {"ETH/USDT", "BTC/USDT"}
    assert all(isinstance(t, str) for _, t in pairs)


def test_process_event_ignores_unmatched():
    assert _feed().process_event(_post_event("just had a great coffee")) == []

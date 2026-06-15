"""Tests for the news RSS connector (offline, synthetic feeds)."""

from sentiment_engine.ingestion.news_rss_feed import NewsRssFeed, parse_feed

RSS_SAMPLE = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Test Feed</title>
  <item>
    <title>Bitcoin surges past resistance as ETF inflows spike</title>
    <guid>item-1</guid>
    <pubDate>Fri, 12 Jun 2026 10:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Cardano upgrade ships on mainnet</title>
    <guid>item-2</guid>
  </item>
  <item>
    <title>Stocks rally on tech earnings</title>
    <guid>item-3</guid>
  </item>
</channel></rss>"""

ATOM_SAMPLE = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>BTC liquidations hit monthly high</title>
    <id>atom-1</id>
    <updated>2026-06-12T10:00:00Z</updated>
  </entry>
</feed>"""


def make_feed():
    return NewsRssFeed(["BTC/USDT", "ADA/USDT"], engine_url="http://test", urls=["http://feed-a"])


def test_parse_rss():
    items = parse_feed(RSS_SAMPLE)
    assert len(items) == 3
    assert items[0].item_id == "item-1"
    assert items[0].published_at is not None


def test_parse_atom():
    items = parse_feed(ATOM_SAMPLE)
    assert len(items) == 1
    assert items[0].title.startswith("BTC liquidations")


def test_parse_garbage_returns_empty():
    assert parse_feed("not xml at all") == []


def test_symbol_matching_aliases_and_word_boundaries():
    feed = make_feed()
    assert feed.match_symbols("Bitcoin breaks out") == ["BTC/USDT"]
    assert feed.match_symbols("Cardano and BTC rally") == ["BTC/USDT", "ADA/USDT"]
    assert feed.match_symbols("Stocks rally on earnings") == []
    assert feed.match_symbols("Nevada adapts regulations") == []  # no substring matches


def test_first_poll_is_baseline_not_flood():
    feed = make_feed()
    assert feed.process_feed_text("http://feed-a", RSS_SAMPLE) == []  # primed, nothing pushed


def test_only_new_items_pushed_after_priming():
    feed = make_feed()
    feed.process_feed_text("http://feed-a", RSS_SAMPLE)  # prime
    updated = RSS_SAMPLE.replace(
        "<guid>item-3</guid>",
        "<guid>item-3</guid></item><item><title>Bitcoin crashes on exchange hack</title><guid>item-4</guid>",
    )
    pushes = feed.process_feed_text("http://feed-a", updated)
    assert pushes == [("BTC/USDT", "Bitcoin crashes on exchange hack")]
    # And never twice:
    assert feed.process_feed_text("http://feed-a", updated) == []


def test_non_matching_new_items_not_pushed():
    feed = make_feed()
    feed.process_feed_text("http://feed-a", RSS_SAMPLE)
    updated = RSS_SAMPLE.replace(
        "<guid>item-3</guid>",
        "<guid>item-3</guid></item><item><title>Gold hits record high</title><guid>item-5</guid>",
    )
    assert feed.process_feed_text("http://feed-a", updated) == []


def test_recent_social_endpoint():
    from fastapi.testclient import TestClient
    from sentiment_engine.api import app

    client = TestClient(app)
    client.post("/ingest/social", json={
        "symbol": "ADA/USDT", "source": "news:test.com",
        "text": "Cardano upgrade ships on mainnet",
    })
    client.post("/ingest/social", json={
        "symbol": "ADA/USDT", "source": "dashboard",
        "text": "ADA breakout, strong buyers",
    })
    data = client.get("/social/ADA-USDT/recent?limit=10").json()
    assert data["symbol"] == "ADA/USDT"
    assert len(data["events"]) >= 2
    newest = data["events"][0]
    assert newest["text"] == "ADA breakout, strong buyers"
    assert newest["sentiment"] is not None
    sources = {event["source"] for event in data["events"]}
    assert "news:test.com" in sources


def test_recent_social_rejects_bad_symbol():
    from fastapi.testclient import TestClient
    from sentiment_engine.api import app

    assert TestClient(app).get("/social/LTC-USDT/recent").status_code == 400


def test_reddit_implied_symbol_for_coin_subreddit():
    from sentiment_engine.ingestion.reddit_feed import RedditFeed

    feed = RedditFeed(["BTC/USDT", "ADA/USDT"], engine_url="http://test")
    feed.process_listing("Bitcoin", {"data": {"children": [
        {"data": {"name": "p1", "title": "old post", "created_utc": 1}},
    ]}})  # prime
    listing = {"data": {"children": [
        {"data": {"name": "p1", "title": "old post", "created_utc": 1}},
        {"data": {"name": "p2", "title": "Just bought my first sats!", "created_utc": 2}},
    ]}}
    # Title never says bitcoin/btc, but r/Bitcoin implies BTC.
    assert feed.process_listing("Bitcoin", listing) == [("BTC/USDT", "Just bought my first sats!")]


def test_generic_subreddit_still_requires_title_match():
    from sentiment_engine.ingestion.reddit_feed import RedditFeed

    feed = RedditFeed(["BTC/USDT", "ADA/USDT"], engine_url="http://test")
    feed.process_listing("CryptoCurrency", {"data": {"children": [
        {"data": {"name": "x1", "title": "old", "created_utc": 1}},
    ]}})
    listing = {"data": {"children": [
        {"data": {"name": "x1", "title": "old", "created_utc": 1}},
        {"data": {"name": "x2", "title": "What are you all holding?", "created_utc": 2}},
    ]}}
    assert feed.process_listing("CryptoCurrency", listing) == []

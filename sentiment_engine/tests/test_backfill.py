"""Tests for the backfill collectors and dedupe guard."""

import time
from decimal import Decimal

from sentiment_engine.ingestion.backfill import collect_news_items, collect_reddit_items
from sentiment_engine.ingestion.news_rss_feed import NewsRssFeed
from sentiment_engine.ingestion.reddit_feed import RedditFeed

NOW = 1_000_000.0


def rss(pubdate_offset_hours):
    from email.utils import formatdate
    return f"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title>Bitcoin rallies hard</title><guid>g1</guid>
    <pubDate>{formatdate(NOW - pubdate_offset_hours * 3600)}</pubDate></item>
  <item><title>Cardano upgrade ships</title><guid>g2</guid>
    <pubDate>{formatdate(NOW - 100 * 3600)}</pubDate></item>
  <item><title>No pubdate bitcoin item</title><guid>g3</guid></item>
</channel></rss>"""


def test_news_backfill_window_and_timestamps():
    matcher = NewsRssFeed(["BTC/USDT", "ADA/USDT"], engine_url="http://x")
    items = collect_news_items(matcher, "https://feed.example.com/rss", rss(2), NOW, hours=48)
    # Only the in-window, dated, symbol-matched item survives.
    assert len(items) == 1
    symbol, source, text, ts = items[0]
    assert symbol == "BTC/USDT"
    assert source == "news:feed.example.com"
    assert abs(ts - (NOW - 2 * 3600)) < 2  # real publish time, not now


def test_reddit_backfill_window():
    matcher = RedditFeed(["BTC/USDT", "ADA/USDT"], engine_url="http://x")
    listing = {"data": {"children": [
        {"data": {"name": "a", "title": "BTC dip incoming", "created_utc": NOW - 3600}},
        {"data": {"name": "b", "title": "BTC ancient post", "created_utc": NOW - 90 * 3600}},
        {"data": {"name": "c", "title": "untitled musings", "created_utc": NOW - 60}},
    ]}}
    # In r/Bitcoin, even an unrelated-sounding title implies BTC (it's the
    # coin's own subreddit); the 90h-old post stays outside the window.
    items = collect_reddit_items(matcher, "Bitcoin", listing, NOW, hours=48)
    assert [(s, t) for s, _, t, _ in items] == [
        ("BTC/USDT", "BTC dip incoming"),
        ("BTC/USDT", "untitled musings"),
    ]

    # A generic subreddit still requires a title match.
    generic = collect_reddit_items(matcher, "CryptoMarkets", listing, NOW, hours=48)
    assert [(s, t) for s, _, t, _ in generic] == [("BTC/USDT", "BTC dip incoming")]


def test_social_history_dedupe(tmp_path):
    from sentiment_engine.storage.social_history import SocialHistory
    from sentiment_engine.types import SocialEvent

    history = SocialHistory(tmp_path / "d.db")
    event = SocialEvent(symbol="BTC/USDT", source="news:x.com", text="Bitcoin rallies",
                        timestamp=NOW, sentiment=Decimal("0.5"), confidence=Decimal("0.8"))
    history.record_event(event)
    history.record_event(event)  # exact duplicate (re-run backfill)
    rows = history.events_between("BTC/USDT", NOW - 10, NOW + 10)
    history.close()
    assert len(rows) == 1

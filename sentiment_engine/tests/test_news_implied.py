"""News feed implied-symbol tagging (coin-specific feeds)."""

from sentiment_engine.ingestion.news_rss_feed import NewsRssFeed

CARDANO_FEED = "https://cointelegraph.com/rss/tag/cardano"
GENERAL_FEED = "https://decrypt.co/feed"


def _feed():
    return NewsRssFeed(
        ["BTC/USDT", "ADA/USDT"],
        engine_url="http://x",
        implied_symbols={CARDANO_FEED: "ADA/USDT", "https://bitcoinmagazine.com/feed": "BTC/USDT"},
    )


def test_headline_match_still_works():
    f = _feed()
    assert f.symbols_for(GENERAL_FEED, "Bitcoin rallies as ETF inflows surge") == ["BTC/USDT"]


def test_implied_symbol_when_headline_omits_coin():
    f = _feed()
    # No "cardano"/"ada" token, but the feed is Cardano-specific -> tagged ADA.
    assert f.symbols_for(CARDANO_FEED, "Hoskinson unveils new governance roadmap") == ["ADA/USDT"]


def test_general_feed_without_match_yields_nothing():
    f = _feed()
    assert f.symbols_for(GENERAL_FEED, "Hoskinson unveils new governance roadmap") == []


def test_explicit_headline_match_overrides_implied():
    f = _feed()
    # Headline names BTC even on the Cardano feed -> follows the headline.
    assert f.symbols_for(CARDANO_FEED, "Bitcoin breaks 70k") == ["BTC/USDT"]


def test_process_feed_text_uses_implied_after_priming():
    f = _feed()
    item = '<rss><channel><item><title>Midnight sidechain goes live</title><guid>g1</guid></item></channel></rss>'
    f.process_feed_text(CARDANO_FEED, item)  # first poll primes, no push
    item2 = '<rss><channel><item><title>Staking rewards updated</title><guid>g2</guid></item></channel></rss>'
    pushes = f.process_feed_text(CARDANO_FEED, item2)
    assert pushes == [("ADA/USDT", "Staking rewards updated")]

"""Coverage for the added SOL, XRP, DOGE, LINK symbols + false-positive guards."""

from sentiment_engine.ingestion.news_rss_feed import NewsRssFeed
from sentiment_engine.ingestion.bluesky_feed import BlueskyFeed
from sentiment_engine.processing.coin_mapper import normalize_symbol

ALL = ["BTC/USDT", "ADA/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", "DOGE/USDT", "LINK/USDT"]


def _news():
    return NewsRssFeed(ALL, engine_url="http://x")


def _bsky():
    return BlueskyFeed(ALL, engine_url="http://x")


def test_new_symbols_normalize():
    assert normalize_symbol("solusdt") == "SOL/USDT"
    assert normalize_symbol("XRP-USDT") == "XRP/USDT"
    assert normalize_symbol("dogeusdt") == "DOGE/USDT"
    assert normalize_symbol("LINK/USDT") == "LINK/USDT"


def test_new_coin_headlines_match():
    m = _news()
    assert m.match_symbols("Solana SOL network sees record volume") == ["SOL/USDT"]
    assert m.match_symbols("XRP rallies as Ripple wins case") == ["XRP/USDT"]
    assert m.match_symbols("Dogecoin jumps on Musk tweet") == ["DOGE/USDT"]
    assert m.match_symbols("Chainlink expands CCIP support") == ["LINK/USDT"]


def test_link_does_not_match_bare_link():
    # bare "link" is everywhere; must only match via "chainlink".
    m = _news()
    assert "LINK/USDT" not in m.match_symbols("click this link to read more")
    assert "LINK/USDT" not in m.match_symbols("the link between price and volume")


def test_bluesky_strict_for_new_coins():
    f = _bsky()
    # cashtag or full name
    assert f.match_symbols("aping into $SOL") == ["SOL/USDT"]
    assert "LINK/USDT" in f.match_symbols("chainlink oracle update")
    # bare "link"/"doge"/"sol" should not trigger on the open firehose
    assert f.match_symbols("here is a link to my blog") == []
    assert f.match_symbols("such doge much wow") == []


def test_all_eight_symbols_supported():
    for s in ALL:
        assert normalize_symbol(s) == s

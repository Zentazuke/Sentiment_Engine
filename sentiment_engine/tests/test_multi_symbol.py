"""Coverage for the added ETH and BNB symbols."""

from sentiment_engine.ingestion.news_rss_feed import NewsRssFeed
from sentiment_engine.processing.coin_mapper import normalize_symbol


def _m():
    return NewsRssFeed(["BTC/USDT", "ADA/USDT", "ETH/USDT", "BNB/USDT"], engine_url="http://x")


def test_new_symbols_normalize():
    assert normalize_symbol("ethusdt") == "ETH/USDT"
    assert normalize_symbol("BNB-USDT") == "BNB/USDT"


def test_eth_headline_matches():
    assert _m().match_symbols("Ethereum gas fees drop after upgrade") == ["ETH/USDT"]
    assert "ETH/USDT" in _m().match_symbols("ETH leads altcoin rally")


def test_bnb_headline_matches():
    assert _m().match_symbols("BNB hits new high on burn news") == ["BNB/USDT"]


def test_binance_exchange_does_not_tag_bnb():
    # The exchange name alone must not be read as the coin.
    assert "BNB/USDT" not in _m().match_symbols("Binance lists a new trading pair")


def test_no_false_positive_substrings():
    # 'tether' must not match 'eth'; 'Canada' must not match 'ada'.
    syms = _m().match_symbols("Tether expands in Canada")
    assert "ETH/USDT" not in syms and "ADA/USDT" not in syms

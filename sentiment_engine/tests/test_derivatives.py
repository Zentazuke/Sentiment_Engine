"""Funding rate + open interest: signal math, feed parsing, storage."""

from sentiment_engine.ingestion.binance_derivatives_feed import BinanceDerivativesFeed
from sentiment_engine.ingestion.binance_lsr_feed import parse_latest
from sentiment_engine.storage.derivatives_history import DerivativesHistory, funding_signal


def test_funding_signal_bounded_and_signed():
    assert funding_signal(0.0) == 0.0
    assert funding_signal(0.0005) == 1.0      # extreme long crowding -> +1
    assert funding_signal(-0.0005) == -1.0    # extreme short crowding -> -1
    assert funding_signal(0.001) == 1.0       # clamped
    assert funding_signal(0.00025) == 0.5
    assert funding_signal(None) is None


def test_parse_latest_funding_and_oi():
    funding_payload = [{"symbol": "BTCUSDT", "fundingRate": "0.00012", "fundingTime": 1}]
    assert parse_latest(funding_payload, "fundingRate") == 0.00012
    oi_payload = [{"symbol": "BTCUSDT", "sumOpenInterest": "1000", "sumOpenInterestValue": "65000000", "timestamp": 1}]
    assert parse_latest(oi_payload, "sumOpenInterestValue") == 65000000.0


def test_feed_constructs_with_symbols():
    f = BinanceDerivativesFeed(["BTC/USDT", "ETH/USDT"], engine_url="http://x")
    assert f.symbols == ["BTC/USDT", "ETH/USDT"]
    assert f.engine_url == "http://x"


def test_storage_record_latest_and_series(tmp_path):
    db = tmp_path / "d.db"
    dh = DerivativesHistory(db)
    dh.record(symbol="BTC/USDT", funding_rate=0.00025, open_interest_usd=65e9, timestamp=100.0)
    dh.record(symbol="BTC/USDT", funding_rate=-0.0005, open_interest_usd=66e9, timestamp=200.0)
    latest = dh.latest("BTC/USDT")
    assert latest["timestamp"] == 200.0
    assert latest["funding_rate"] == -0.0005
    assert latest["open_interest_usd"] == 66e9
    assert latest["funding_signal"] == -1.0
    assert dh.latest("ETH/USDT") is None
    series = dh.funding_series_between("BTC/USDT", 50.0, 150.0)
    assert series == [(100.0, 0.5)]

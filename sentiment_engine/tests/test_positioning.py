"""Long/short positioning: signal math, feed parsing, storage."""

from sentiment_engine.ingestion.binance_lsr_feed import futures_symbol, parse_latest
from sentiment_engine.storage.positioning_history import (
    PositioningHistory,
    long_fraction_signal,
    ratio_to_long_fraction,
)


def test_ratio_to_long_fraction():
    assert ratio_to_long_fraction(1.0) == 0.5            # equal longs/shorts
    assert abs(ratio_to_long_fraction(3.0) - 0.75) < 1e-9
    assert ratio_to_long_fraction(None) is None
    assert ratio_to_long_fraction(-1) is None


def test_long_fraction_signal():
    assert long_fraction_signal(0.5) == 0.0
    assert long_fraction_signal(1.0) == 1.0
    assert long_fraction_signal(0.0) == -1.0
    assert long_fraction_signal(None) is None


def test_futures_symbol():
    assert futures_symbol("BTC/USDT") == "BTCUSDT"
    assert futures_symbol("ETH/USDT") == "ETHUSDT"


def test_parse_latest_takes_last_element():
    payload = [
        {"longShortRatio": "1.0", "timestamp": 1},
        {"longShortRatio": "2.5", "timestamp": 2},
    ]
    assert parse_latest(payload, "longShortRatio") == 2.5
    assert parse_latest([], "longShortRatio") is None
    assert parse_latest("nope", "longShortRatio") is None
    assert parse_latest([{"x": 1}], "longShortRatio") is None


def test_storage_record_and_latest(tmp_path):
    db = tmp_path / "p.db"
    ph = PositioningHistory(db)
    ph.record(symbol="BTC/USDT", global_account_ratio=3.0, top_account_ratio=2.0,
              top_position_ratio=1.5, taker_ratio=1.1, timestamp=100.0)
    ph.record(symbol="BTC/USDT", global_account_ratio=1.0, top_account_ratio=1.0,
              top_position_ratio=1.0, taker_ratio=1.0, timestamp=200.0)
    latest = ph.latest("BTC/USDT")
    assert latest["timestamp"] == 200.0
    assert latest["global_account_ratio"] == 1.0
    assert latest["signal"] == 0.0  # ratio 1.0 -> 50% long -> neutral
    assert ph.latest("ETH/USDT") is None


def test_storage_series(tmp_path):
    ph = PositioningHistory(tmp_path / "p.db")
    ph.record(symbol="BTC/USDT", global_account_ratio=3.0, top_account_ratio=None,
              top_position_ratio=None, taker_ratio=None, timestamp=150.0)
    series = ph.series_between("BTC/USDT", 100.0, 200.0)
    assert len(series) == 1
    assert series[0][0] == 150.0
    assert abs(series[0][1] - 0.5) < 1e-9  # ratio 3 -> 75% long -> signal 0.5

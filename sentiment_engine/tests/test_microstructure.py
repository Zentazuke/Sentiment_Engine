"""Tests for microstructure scoring functions, feeds (offline), and API."""

import json
import time

import pytest

from sentiment_engine.signals.microstructure import (
    Trade,
    book_imbalance,
    buy_sell_volume,
    depth_quote_value,
    momentum_pct,
    relative_volume,
    spread_bps,
    trade_imbalance,
    trades_in_window,
    volatility_bps,
    vwap,
    vwap_distance_bps,
)


def make_trade(price=100.0, qty=1.0, ts=0.0, buy=True):
    return Trade(price=price, quantity=qty, timestamp=ts, is_aggressive_buy=buy)


# --- pure functions -----------------------------------------------------

def test_trade_imbalance_all_buys():
    assert trade_imbalance(10.0, 0.0) == 1.0


def test_trade_imbalance_all_sells():
    assert trade_imbalance(0.0, 10.0) == -1.0


def test_trade_imbalance_balanced():
    assert trade_imbalance(5.0, 5.0) == 0.0


def test_trade_imbalance_no_volume_is_none():
    assert trade_imbalance(0.0, 0.0) is None


def test_buy_sell_volume_splits_by_aggressor():
    trades = [make_trade(qty=2.0, buy=True), make_trade(qty=3.0, buy=False)]
    assert buy_sell_volume(trades) == (2.0, 3.0)


def test_trades_in_window_bounds():
    trades = [make_trade(ts=t) for t in (1.0, 5.0, 10.0)]
    selected = trades_in_window(trades, 1.0, 10.0)  # start exclusive, end inclusive
    assert [t.timestamp for t in selected] == [5.0, 10.0]


def test_relative_volume_ratio():
    assert relative_volume(30.0, 10.0) == 3.0


def test_relative_volume_no_baseline_is_none():
    assert relative_volume(30.0, 0.0) is None


def test_momentum_positive():
    trades = [make_trade(price=100.0, ts=0), make_trade(price=101.0, ts=10)]
    assert momentum_pct(trades) == pytest.approx(1.0)


def test_momentum_insufficient_data_is_none():
    assert momentum_pct([make_trade()]) is None


def test_volatility_flat_prices_is_zero():
    trades = [make_trade(price=100.0, ts=float(i)) for i in range(10)]
    assert volatility_bps(trades) == pytest.approx(0.0)


def test_volatility_moving_prices_positive():
    prices = [100.0, 101.0, 99.5, 100.8, 99.9, 100.4]
    trades = [make_trade(price=p, ts=float(i)) for i, p in enumerate(prices)]
    assert volatility_bps(trades) > 0


def test_volatility_insufficient_data_is_none():
    assert volatility_bps([make_trade(ts=0.0), make_trade(ts=1.0)]) is None


def test_vwap_weighted():
    trades = [make_trade(price=100.0, qty=1.0), make_trade(price=200.0, qty=3.0)]
    assert vwap(trades) == pytest.approx(175.0)


def test_vwap_distance_above():
    assert vwap_distance_bps(101.0, 100.0) == pytest.approx(100.0)


def test_vwap_distance_none_when_no_vwap():
    assert vwap_distance_bps(101.0, None) is None


def test_spread_bps():
    assert spread_bps(99.995, 100.005) == pytest.approx(1.0, rel=1e-3)


def test_spread_crossed_book_is_none():
    assert spread_bps(100.0, 99.0) is None


def test_depth_quote_value_band():
    levels = [(100.0, 1.0), (99.8, 2.0), (95.0, 100.0)]  # last is outside band
    value = depth_quote_value(levels, 100.0, band_pct=0.25)
    assert value == pytest.approx(100.0 * 1.0 + 99.8 * 2.0)


def test_book_imbalance_bid_heavy():
    assert book_imbalance(75.0, 25.0) == pytest.approx(0.5)


def test_book_imbalance_none_when_missing():
    assert book_imbalance(None, 25.0) is None


# --- feeds (offline, synthetic messages) --------------------------------

def _agg_trade_message(price, qty, ts_ms, buyer_is_maker, stream="btcusdt@aggTrade"):
    return json.dumps({
        "stream": stream,
        "data": {"p": str(price), "q": str(qty), "T": ts_ms, "m": buyer_is_maker},
    })


def test_trade_feed_metrics_from_messages():
    from sentiment_engine.ingestion.binance_trade_feed import BinanceTradeFeed

    feed = BinanceTradeFeed(["BTC/USDT"])
    now = time.time()
    # 3 aggressive buys then 1 aggressive sell within the last 10s.
    for i, maker in enumerate([False, False, False, True]):
        feed.handle_message(_agg_trade_message(100.0 + i, 1.0, int((now - 8 + i) * 1000), maker))
    metrics = feed.metrics("BTC/USDT", now=now)
    assert metrics["last_price"] == pytest.approx(103.0)
    assert metrics["buy_volume_10s"] == pytest.approx(3.0)
    assert metrics["sell_volume_10s"] == pytest.approx(1.0)
    assert metrics["trade_imbalance_10s"] == pytest.approx(0.5)
    assert metrics["trade_count_60s"] == 4
    assert metrics["relative_volume"] is None  # warmup not reached


def test_trade_feed_ignores_garbage():
    from sentiment_engine.ingestion.binance_trade_feed import BinanceTradeFeed

    feed = BinanceTradeFeed(["BTC/USDT"])
    feed.handle_message("not json")
    feed.handle_message(json.dumps({"stream": "btcusdt@aggTrade", "data": {}}))
    assert feed.metrics("BTC/USDT") == {}


def test_book_feed_metrics_from_message():
    from sentiment_engine.ingestion.binance_orderbook_feed import BinanceOrderBookFeed

    feed = BinanceOrderBookFeed(["BTC/USDT"])
    feed.handle_message(json.dumps({
        "stream": "btcusdt@depth20@100ms",
        "data": {
            "bids": [["100.00", "3.0"], ["99.90", "2.0"]],
            "asks": [["100.10", "1.0"], ["100.20", "1.0"]],
        },
    }))
    metrics = feed.metrics("BTC/USDT")
    assert metrics["spread_bps"] == pytest.approx(10.0, rel=1e-2)
    assert metrics["bid_depth_quote"] > metrics["ask_depth_quote"]
    assert metrics["book_imbalance"] > 0


def test_book_feed_old_book_reports_nothing():
    from sentiment_engine.ingestion.binance_orderbook_feed import BinanceOrderBookFeed

    feed = BinanceOrderBookFeed(["BTC/USDT"])
    feed.handle_message(json.dumps({
        "stream": "btcusdt@depth20@100ms",
        "data": {"bids": [["100.0", "1.0"]], "asks": [["100.1", "1.0"]]},
    }))
    assert feed.metrics("BTC/USDT", now=time.time() + 60) == {}


# --- run_live payload assembly -------------------------------------------

def test_build_payload_merges_and_drops_none():
    from sentiment_engine.ingestion.binance_orderbook_feed import BinanceOrderBookFeed
    from sentiment_engine.ingestion.binance_trade_feed import BinanceTradeFeed
    from sentiment_engine.ingestion.run_live import build_payload

    trade_feed = BinanceTradeFeed(["BTC/USDT"])
    book_feed = BinanceOrderBookFeed(["BTC/USDT"])
    assert build_payload("BTC/USDT", trade_feed, book_feed) is None  # no data yet

    now = time.time()
    trade_feed.handle_message(_agg_trade_message(100.0, 1.0, int((now - 2) * 1000), False))
    trade_feed.handle_message(_agg_trade_message(100.5, 1.0, int((now - 1) * 1000), False))
    payload = build_payload("BTC/USDT", trade_feed, book_feed, now=now)
    assert payload["symbol"] == "BTC/USDT"
    assert payload["last_price"] == pytest.approx(100.5)
    assert "spread_bps" not in payload  # book has no data; None values dropped


# --- engine state + API ---------------------------------------------------

def test_state_store_staleness():
    from sentiment_engine.aggregation.state_store import StateStore
    from sentiment_engine.types import MicrostructureSnapshot

    store = StateStore()
    assert store.microstructure("BTC/USDT") == (None, True)

    now = time.time()
    snap = MicrostructureSnapshot(symbol="BTC/USDT", computed_at=now, last_price=100.0)
    store.add_microstructure(snap, received_at=now)
    fresh, stale = store.microstructure("BTC/USDT", now=now + 1)
    assert fresh is not None and not stale
    _, stale_later = store.microstructure("BTC/USDT", now=now + 60)
    assert stale_later


def test_api_ingest_and_snapshot_microstructure():
    from fastapi.testclient import TestClient
    from sentiment_engine.api import app

    client = TestClient(app)
    response = client.post("/ingest/microstructure", json={
        "symbol": "BTC/USDT",
        "last_price": 104000.5,
        "trade_imbalance_10s": 0.4,
        "spread_bps": 0.8,
        "book_imbalance": 0.2,
    })
    assert response.status_code == 200
    assert response.json()["accepted"] is True

    snapshot = client.get("/snapshot/BTC-USDT").json()
    micro = snapshot["microstructure"]
    assert micro is not None
    assert micro["last_price"] == pytest.approx(104000.5)
    assert micro["trade_imbalance_10s"] == pytest.approx(0.4)
    assert micro["stale"] is False
    assert micro["trade_imbalance_30s"] is None  # not sent -> unavailable


def test_api_microstructure_rejects_unsupported_symbol():
    from fastapi.testclient import TestClient
    from sentiment_engine.api import app

    client = TestClient(app)
    response = client.post("/ingest/microstructure", json={"symbol": "LTC/USDT"})
    assert response.status_code == 400

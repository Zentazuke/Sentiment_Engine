"""Tests for the layered decision (micro primary, sentiment modifier),
regime gates, micro quality, history recorder, and outcome tracking."""

import time
from decimal import Decimal

import pytest

from sentiment_engine.signals.decision import evaluate_trade_signal
from sentiment_engine.signals.micro_quality import micro_direction_score, micro_quality_score
from sentiment_engine.types import (
    MicrostructureSnapshot,
    SentimentAction,
    SentimentSnapshot,
    TradeDirection,
)


def make_sentiment_snapshot(**overrides):
    defaults = dict(
        symbol="BTC/USDT",
        window_seconds=60,
        message_count=0,
        previous_message_count=0,
        average_sentiment=Decimal("0"),
        previous_average_sentiment=Decimal("0"),
        sentiment_velocity=Decimal("0"),
        attention_spike=Decimal("0"),
        price_change_pct=Decimal("0"),
        data_quality=Decimal("0"),
    )
    defaults.update(overrides)
    return SentimentSnapshot(**defaults)


def make_micro(**overrides):
    defaults = dict(
        symbol="BTC/USDT",
        computed_at=time.time(),
        last_price=100000.0,
        trade_imbalance_10s=0.0,
        trade_imbalance_30s=0.0,
        trade_imbalance_60s=0.0,
        book_imbalance=0.0,
        spread_bps=1.0,
        volatility_bps_30s=5.0,
        bid_depth_quote=500000.0,
        ask_depth_quote=500000.0,
        trade_count_60s=100,
    )
    defaults.update(overrides)
    return MicrostructureSnapshot(**defaults)


# --- micro quality ---------------------------------------------------------

def test_quality_zero_when_missing_or_stale():
    assert micro_quality_score(None, stale=True) == 0.0
    assert micro_quality_score(make_micro(), stale=True) == 0.0


def test_quality_full_with_all_metrics():
    assert micro_quality_score(make_micro(), stale=False) == 1.0


def test_quality_reduced_with_low_trade_count():
    quality = micro_quality_score(make_micro(trade_count_60s=3), stale=False)
    assert 0.0 < quality < 1.0


def test_direction_score_requires_trade_flow():
    micro = make_micro(trade_imbalance_10s=None, trade_imbalance_30s=None, book_imbalance=0.9)
    assert micro_direction_score(micro) is None  # book alone is spoofable


def test_direction_score_blends():
    micro = make_micro(trade_imbalance_10s=1.0, trade_imbalance_30s=1.0, book_imbalance=1.0)
    assert micro_direction_score(micro) == pytest.approx(1.0)


# --- layered decision -------------------------------------------------------

def test_micro_buy_pressure_confirms_long():
    micro = make_micro(trade_imbalance_10s=0.6, trade_imbalance_30s=0.5, book_imbalance=0.4)
    result = evaluate_trade_signal(
        snapshot=make_sentiment_snapshot(), direction=TradeDirection.LONG,
        bot_confidence=Decimal("0.7"), micro=micro, micro_stale=False,
    )
    assert result.action is SentimentAction.CONFIRM
    assert result.safe_to_use


def test_micro_sell_pressure_vetoes_long():
    micro = make_micro(trade_imbalance_10s=-0.7, trade_imbalance_30s=-0.6, book_imbalance=-0.5)
    result = evaluate_trade_signal(
        snapshot=make_sentiment_snapshot(), direction=TradeDirection.LONG,
        bot_confidence=Decimal("0.7"), micro=micro, micro_stale=False,
    )
    assert result.action is SentimentAction.VETO


def test_micro_sell_pressure_confirms_short():
    micro = make_micro(trade_imbalance_10s=-0.6, trade_imbalance_30s=-0.5, book_imbalance=-0.4)
    result = evaluate_trade_signal(
        snapshot=make_sentiment_snapshot(), direction=TradeDirection.SHORT,
        bot_confidence=Decimal("0.7"), micro=micro, micro_stale=False,
    )
    assert result.action is SentimentAction.CONFIRM


def test_balanced_micro_is_neutral():
    result = evaluate_trade_signal(
        snapshot=make_sentiment_snapshot(), direction=TradeDirection.LONG,
        bot_confidence=Decimal("0.7"), micro=make_micro(), micro_stale=False,
    )
    assert result.action is SentimentAction.NEUTRAL


def test_wide_spread_gate_vetoes_even_with_buy_pressure():
    micro = make_micro(trade_imbalance_10s=0.9, trade_imbalance_30s=0.9, book_imbalance=0.9,
                       spread_bps=25.0)
    result = evaluate_trade_signal(
        snapshot=make_sentiment_snapshot(), direction=TradeDirection.LONG,
        bot_confidence=Decimal("0.7"), micro=micro, micro_stale=False,
    )
    assert result.action is SentimentAction.VETO
    assert "spread" in result.reason.lower()


def test_volatility_burst_blocks_confirm_but_not_veto():
    bullish = make_micro(trade_imbalance_10s=0.8, trade_imbalance_30s=0.7, book_imbalance=0.6,
                         volatility_bps_30s=80.0)
    result = evaluate_trade_signal(
        snapshot=make_sentiment_snapshot(), direction=TradeDirection.LONG,
        bot_confidence=Decimal("0.7"), micro=bullish, micro_stale=False,
    )
    assert result.action is SentimentAction.NEUTRAL  # confirm blocked by burst

    result_short = evaluate_trade_signal(
        snapshot=make_sentiment_snapshot(), direction=TradeDirection.SHORT,
        bot_confidence=Decimal("0.7"), micro=bullish, micro_stale=False,
    )
    assert result_short.action is SentimentAction.VETO  # veto still allowed


def test_stale_micro_falls_back_to_sentiment_only():
    micro = make_micro(trade_imbalance_10s=0.9, trade_imbalance_30s=0.9, book_imbalance=0.9)
    result = evaluate_trade_signal(
        snapshot=make_sentiment_snapshot(), direction=TradeDirection.LONG,
        bot_confidence=Decimal("0.7"), micro=micro, micro_stale=True,
    )
    assert result.action is SentimentAction.NEUTRAL
    assert not result.safe_to_use  # no usable data at all


def test_sentiment_nudges_confidence_but_never_flips_action():
    micro = make_micro(trade_imbalance_10s=0.6, trade_imbalance_30s=0.5, book_imbalance=0.4)
    bearish_sentiment = make_sentiment_snapshot(
        data_quality=Decimal("0.9"), sentiment_velocity=Decimal("-0.50"),
        attention_spike=Decimal("0.9"), message_count=30,
    )
    result = evaluate_trade_signal(
        snapshot=bearish_sentiment, direction=TradeDirection.LONG,
        bot_confidence=Decimal("0.7"), micro=micro, micro_stale=False,
    )
    assert result.action is SentimentAction.CONFIRM  # micro decision stands
    assert result.confidence_modifier == Decimal("1.00")  # 1.05 - 0.05 sentiment penalty


def test_aligned_sentiment_boosts_confirm_modifier():
    micro = make_micro(trade_imbalance_10s=0.6, trade_imbalance_30s=0.5, book_imbalance=0.4)
    bullish_sentiment = make_sentiment_snapshot(
        data_quality=Decimal("0.9"), sentiment_velocity=Decimal("0.50"),
        attention_spike=Decimal("0.9"), message_count=30,
    )
    result = evaluate_trade_signal(
        snapshot=bullish_sentiment, direction=TradeDirection.LONG,
        bot_confidence=Decimal("0.7"), micro=micro, micro_stale=False,
    )
    assert result.action is SentimentAction.CONFIRM
    assert result.confidence_modifier == Decimal("1.10")


# --- history recorder --------------------------------------------------------

def test_micro_history_records_and_reads(tmp_path):
    from sentiment_engine.storage.micro_history import MicroHistory
    import sqlite3

    history = MicroHistory(tmp_path / "history.db")
    history.record(make_micro(last_price=123.0), received_at=1000.0)
    history.record(make_micro(last_price=124.0), received_at=1001.0)
    conn = sqlite3.connect(tmp_path / "history.db")
    rows = conn.execute("SELECT symbol, last_price FROM micro_history ORDER BY received_at").fetchall()
    conn.close()
    history.close()
    assert rows == [("BTC/USDT", 123.0), ("BTC/USDT", 124.0)]


def test_micro_stats_percentile():
    from sentiment_engine.storage.micro_stats import percentile

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 50) == 3.0
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 5.0


# --- outcome tracking ---------------------------------------------------------

def test_outcome_tracker_fills_due_horizons():
    from sentiment_engine.storage.outcome_tracker import OutcomeTracker

    written = []
    tracker = OutcomeTracker(
        write_outcome=lambda row, horizon, price: written.append((row, horizon, price)),
        get_price=lambda symbol: 105.5,
    )
    tracker.register(row_id=7, symbol="BTC/USDT", evaluated_at=1000.0)
    assert tracker.process_due(now=1000.0 + 29) == 0       # nothing due yet
    assert tracker.process_due(now=1000.0 + 31) == 1       # 30s due
    assert written == [(7, 30, 105.5)]
    assert tracker.process_due(now=1000.0 + 301) == 2      # 120s + 300s due
    assert tracker.pending_count == 0


def test_outcome_tracker_records_null_when_too_late():
    from sentiment_engine.storage.outcome_tracker import OutcomeTracker

    written = []
    tracker = OutcomeTracker(
        write_outcome=lambda row, horizon, price: written.append((row, horizon, price)),
        get_price=lambda symbol: 105.5,
    )
    tracker.register(row_id=1, symbol="BTC/USDT", evaluated_at=1000.0)
    tracker.process_due(now=1000.0 + 30 + 60)  # 60s past the 30s horizon
    assert written[0] == (1, 30, None)


def test_journal_outcome_roundtrip(tmp_path):
    import sqlite3
    from sentiment_engine.storage.sqlite_journal import SentimentJournal
    from sentiment_engine.signals.decision import evaluate_trade_signal

    journal = SentimentJournal(tmp_path / "journal.db")
    result = evaluate_trade_signal(
        snapshot=make_sentiment_snapshot(), direction=TradeDirection.LONG,
        bot_confidence=Decimal("0.7"),
    )
    row_id = journal.log_evaluation(
        evaluation=result, direction=TradeDirection.LONG,
        bot_confidence=Decimal("0.7"), timestamp=1000.0, price_at_eval=100.0,
    )
    assert row_id > 0
    journal.fill_outcome(row_id, 30, 101.5)
    with pytest.raises(ValueError):
        journal.fill_outcome(row_id, 999, 101.5)
    conn = sqlite3.connect(tmp_path / "journal.db")
    row = conn.execute(
        "SELECT price_at_eval, price_after_30s, price_after_120s FROM evaluations WHERE id = ?",
        (row_id,),
    ).fetchone()
    conn.close()
    journal.close()
    assert row == (100.0, 101.5, None)


# --- health endpoint -----------------------------------------------------------

def test_health_reports_feed_freshness():
    from fastapi.testclient import TestClient
    from sentiment_engine.api import app, STATE

    client = TestClient(app)
    feeds = client.get("/health").json()["feeds"]
    assert "BTC-USDT" in feeds and "ADA-USDT" in feeds
    STATE.add_microstructure(make_micro(), received_at=time.time())
    feeds = client.get("/health").json()["feeds"]
    assert feeds["BTC-USDT"]["live"] is True
    assert feeds["BTC-USDT"]["stale"] is False

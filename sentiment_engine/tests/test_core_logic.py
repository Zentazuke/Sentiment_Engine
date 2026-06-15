from decimal import Decimal

from sentiment_engine.processing.sentiment_fast import score_text
from sentiment_engine.signals.attention import attention_spike_score
from sentiment_engine.signals.decision import evaluate_trade_signal
from sentiment_engine.types import SentimentSnapshot, TradeDirection, SentimentAction


def test_fast_sentiment_positive():
    score, confidence = score_text("BTC breakout looks strong, massive pump")
    assert score > 0
    assert confidence > 0


def test_attention_spike_normalizes():
    assert attention_spike_score(30, 10) == Decimal("1.0000")
    assert attention_spike_score(10, 10) == Decimal("0.0000")


def test_long_confirm():
    snapshot = SentimentSnapshot(
        symbol="BTC/USDT",
        window_seconds=60,
        message_count=20,
        previous_message_count=8,
        average_sentiment=Decimal("0.50"),
        previous_average_sentiment=Decimal("0.10"),
        sentiment_velocity=Decimal("0.40"),
        attention_spike=Decimal("0.75"),
        price_change_pct=Decimal("0.20"),
        data_quality=Decimal("0.90"),
    )
    result = evaluate_trade_signal(snapshot=snapshot, direction=TradeDirection.LONG, bot_confidence=Decimal("0.70"))
    assert result.action is SentimentAction.CONFIRM


def test_short_veto_on_bullish_velocity():
    snapshot = SentimentSnapshot(
        symbol="BTC/USDT",
        window_seconds=60,
        message_count=20,
        previous_message_count=8,
        average_sentiment=Decimal("0.50"),
        previous_average_sentiment=Decimal("0.10"),
        sentiment_velocity=Decimal("0.40"),
        attention_spike=Decimal("0.75"),
        price_change_pct=Decimal("0.20"),
        data_quality=Decimal("0.90"),
    )
    result = evaluate_trade_signal(snapshot=snapshot, direction=TradeDirection.SHORT, bot_confidence=Decimal("0.70"))
    assert result.action is SentimentAction.VETO

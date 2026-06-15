"""Optional transformer scorer: label mapping + dispatcher fallback."""

from decimal import Decimal

from sentiment_engine.processing import sentiment_transformer as st
from sentiment_engine.processing import scoring


def test_score_from_distribution_bullish():
    s, conf = st.score_from_distribution([
        {"label": "Bullish", "score": 0.8}, {"label": "Neutral", "score": 0.15}, {"label": "Bearish", "score": 0.05},
    ])
    assert s > 0.7 and 0.7 < conf <= 1.0


def test_score_from_distribution_bearish():
    s, _ = st.score_from_distribution([
        {"label": "negative", "score": 0.9}, {"label": "neutral", "score": 0.07}, {"label": "positive", "score": 0.03},
    ])
    assert s < -0.8


def test_score_from_distribution_neutral_near_zero():
    s, _ = st.score_from_distribution([
        {"label": "Bullish", "score": 0.45}, {"label": "Bearish", "score": 0.45}, {"label": "Neutral", "score": 0.10},
    ])
    assert abs(s) < 1e-9


def test_dispatcher_uses_lexicon_when_disabled(monkeypatch):
    # Transformer off -> identical to the lexicon.
    monkeypatch.setattr(scoring, "transformer_enabled", lambda: False)
    from sentiment_engine.processing.sentiment_fast import score_text as lex
    assert scoring.score_text("BTC breakout, bulls in control") == lex("BTC breakout, bulls in control")


def test_dispatcher_falls_back_when_transformer_unavailable(monkeypatch):
    # Enabled, but inference returns None (model unavailable) -> lexicon result.
    monkeypatch.setattr(scoring, "transformer_enabled", lambda: True)
    monkeypatch.setattr(scoring, "score_with_transformer", lambda text: None)
    from sentiment_engine.processing.sentiment_fast import score_text as lex
    assert scoring.score_text("massive dump incoming") == lex("massive dump incoming")


def test_dispatcher_uses_transformer_when_available(monkeypatch):
    monkeypatch.setattr(scoring, "transformer_enabled", lambda: True)
    monkeypatch.setattr(scoring, "score_with_transformer", lambda text: (Decimal("0.7"), Decimal("0.9")))
    assert scoring.score_text("anything") == (Decimal("0.7"), Decimal("0.9"))

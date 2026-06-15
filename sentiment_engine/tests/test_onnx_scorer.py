"""ONNX-int8 scorer: logits mapping + dispatcher priority/fallback."""

from decimal import Decimal

from sentiment_engine.processing import sentiment_onnx as onx
from sentiment_engine.processing import scoring


def test_softmax_sums_to_one():
    p = onx._softmax([2.0, 1.0, 0.1])
    assert abs(sum(p) - 1.0) < 1e-9
    assert p[0] > p[1] > p[2]


def test_logits_to_score_bullish():
    # labels in CryptoBERT order: bearish, neutral, bullish
    s, conf = onx.logits_to_score([0.1, 0.2, 4.0], ["bearish", "neutral", "bullish"])
    assert s > 0.7 and 0.0 < conf <= 1.0


def test_logits_to_score_bearish():
    s, _ = onx.logits_to_score([5.0, 0.0, -1.0], ["bearish", "neutral", "bullish"])
    assert s < -0.7


def test_logits_to_score_neutral_near_zero():
    s, _ = onx.logits_to_score([1.0, 5.0, 1.0], ["bearish", "neutral", "bullish"])
    assert abs(s) < 0.2


def test_onnx_enabled_requires_both_paths(monkeypatch):
    monkeypatch.setattr(onx, "ONNX_MODEL_PATH", "")
    monkeypatch.setattr(onx, "ONNX_TOKENIZER_PATH", "")
    assert onx.onnx_enabled() is False
    monkeypatch.setattr(onx, "ONNX_MODEL_PATH", "m.onnx")
    assert onx.onnx_enabled() is False  # tokenizer still missing
    monkeypatch.setattr(onx, "ONNX_TOKENIZER_PATH", "t.json")
    assert onx.onnx_enabled() is True


def test_dispatcher_prefers_onnx(monkeypatch):
    monkeypatch.setattr(scoring, "onnx_enabled", lambda: True)
    monkeypatch.setattr(scoring, "score_with_onnx", lambda t: (Decimal("0.6"), Decimal("0.9")))
    # transformer must NOT be consulted when onnx returns a result
    monkeypatch.setattr(scoring, "transformer_enabled", lambda: (_ for _ in ()).throw(AssertionError("should not be called")))
    assert scoring.score_text("x") == (Decimal("0.6"), Decimal("0.9"))


def test_dispatcher_falls_back_when_onnx_unavailable(monkeypatch):
    monkeypatch.setattr(scoring, "onnx_enabled", lambda: True)
    monkeypatch.setattr(scoring, "score_with_onnx", lambda t: None)   # model unavailable
    monkeypatch.setattr(scoring, "transformer_enabled", lambda: False)
    from sentiment_engine.processing.sentiment_fast import score_text as lex
    assert scoring.score_text("massive dump incoming") == lex("massive dump incoming")

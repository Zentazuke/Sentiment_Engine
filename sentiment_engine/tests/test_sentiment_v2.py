"""Tests for sentiment scorer v2: negation, intensifiers, hedging, emojis."""

from decimal import Decimal

from sentiment_engine.processing.sentiment_fast import score_text


def s(text):
    score, confidence = score_text(text)
    return float(score), float(confidence)


# --- basic polarity ----------------------------------------------------------

def test_clearly_bullish():
    score, conf = s("BTC breakout looks strong, massive pump")
    assert score > 0.5 and conf > 0.4


def test_clearly_bearish():
    score, conf = s("Panic selloff, support lost, crash incoming")
    assert score < -0.5 and conf > 0.4


def test_no_signal_words_neutral_low_confidence():
    score, conf = s("The weather is nice today")
    assert score == 0.0 and conf <= 0.25


def test_empty_text():
    assert s("") == (0.0, 0.0)


# --- negation ----------------------------------------------------------------

def test_negation_flips_bullish():
    positive, _ = s("this looks bullish")
    negated, _ = s("this is not bullish at all")
    assert positive > 0 > negated


def test_fake_acts_as_negator():
    score, _ = s("fake pump, do not trust it")
    assert score < 0


def test_no_breakout_negative():
    score, _ = s("no breakout here")
    assert score < 0


# --- intensifiers -------------------------------------------------------------

def test_intensifier_amplifies():
    plain, _ = s("BTC dumping")
    intense, _ = s("BTC dumping hard")
    assert intense < plain < 0


def test_intensifier_before_word():
    plain, _ = s("a pump")
    intense, _ = s("a massive pump")
    assert intense > plain > 0


# --- hedging & uncertainty -----------------------------------------------------

def test_hedge_dampens_score_and_confidence():
    firm_score, firm_conf = s("BTC will breakout, bulls in control")
    hedged_score, hedged_conf = s("BTC might breakout maybe, bulls in control")
    assert 0 < hedged_score < firm_score
    assert hedged_conf < firm_conf


def test_question_reduces_confidence():
    _, conf_plain = s("is this a pump")
    _, conf_question = s("is this a pump?")
    assert conf_question < conf_plain


def test_mixed_signals_lower_confidence():
    _, pure_conf = s("pump pump rally")
    _, mixed_conf = s("pump but also crash and rally then dump")
    assert mixed_conf < pure_conf


# --- phrases & emojis -----------------------------------------------------------

def test_phrase_beats_component_words():
    score, _ = s("rug pull confirmed")
    assert score < -0.5


def test_support_lost_vs_support_held():
    lost, _ = s("support lost")
    held, _ = s("support held")
    assert lost < 0 < held


def test_bull_trap_negative_despite_bull():
    score, _ = s("classic bull trap forming")
    assert score < 0


def test_rocket_emoji_bullish():
    score, _ = s("BTC 🚀🚀")
    assert score > 0


def test_chart_down_emoji_bearish():
    score, _ = s("ADA 📉")
    assert score < 0


# --- dashboard scenario sanity (the React tester depends on these directions) ---

def test_scenario_phrases_have_expected_sign():
    bullish = [
        "BTC breakout looks strong, bulls taking control",
        "BTC pumping hard, support held perfectly",
        "Massive squeeze incoming, buyers stepping in",
        "Clean reclaim, momentum looks bullish",
    ]
    bearish = [
        "BTC dumping hard, support lost",
        "Panic selloff, longs getting rekt",
        "Bearish rejection, buyers disappeared",
        "Crash vibes, this looks ugly",
        "Weak bounce, likely more downside",
        "Another rejection, bearish continuation likely",
    ]
    for text in bullish:
        assert s(text)[0] > 0, text
    for text in bearish:
        assert s(text)[0] < 0, text


def test_bounds_and_types():
    score, conf = score_text("massive insane pump rally surge moon 🚀")
    assert isinstance(score, Decimal) and isinstance(conf, Decimal)
    assert Decimal("-1") <= score <= Decimal("1")
    assert Decimal("0") <= conf <= Decimal("1")

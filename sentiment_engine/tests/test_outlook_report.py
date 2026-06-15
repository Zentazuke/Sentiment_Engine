"""Tests for the outlook validation math (pure functions, synthetic data)."""

import pytest

from sentiment_engine.storage.outlook_report import NEUTRAL_BAND, evaluate_horizon, pearson


def test_pearson_perfect_positive():
    assert pearson([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_pearson_insufficient():
    assert pearson([1, 2], [1, 2]) is None


def test_perfect_predictor_hit_rate_one():
    # score +0.5 -> price up 1%; score -0.5 -> price down 1%
    outlooks = [(t, 0.5 if t % 2 == 0 else -0.5, 0.9, 100.0) for t in range(10)]
    prices = {t + 3600: 101.0 if t % 2 == 0 else 99.0 for t in range(10)}
    result = evaluate_horizon(outlooks, lambda ts: prices.get(ts), 3600)
    assert result["hit_rate"] == 1.0
    assert result["correlation"] == pytest.approx(1.0)
    assert result["mean_return_when_bullish_pct"] == pytest.approx(1.0)
    assert result["mean_return_when_bearish_pct"] == pytest.approx(-1.0)


def test_neutral_scores_excluded_from_calls():
    outlooks = [(0, NEUTRAL_BAND / 2, 0.9, 100.0)]
    result = evaluate_horizon(outlooks, lambda ts: 101.0, 3600)
    assert result["samples"] == 1
    assert result["directional_calls"] == 0
    assert result["hit_rate"] is None


def test_missing_prices_skipped():
    outlooks = [(0, 0.5, 0.9, 100.0), (10, 0.5, 0.9, None)]
    result = evaluate_horizon(outlooks, lambda ts: None, 3600)
    assert result["samples"] == 0

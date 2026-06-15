"""Multi-feature logistic calibration: features, fit, adopt-gate, integration."""

import json

from sentiment_engine.signals.calibration import CalibrationModel, PriceSeries, train
from sentiment_engine.signals.logistic_model import (
    LogisticModel,
    N_FEATURES,
    extract_logistic_features,
    train_logistic,
)
from sentiment_engine.signals.outlook import composite_outlook


def test_feature_vector_shape_and_order():
    horizons = [
        {"horizon_seconds": 3600, "average_sentiment": 0.1, "trend": 0.2},
        {"horizon_seconds": 21600, "average_sentiment": 0.3, "trend": 0.4},
        {"horizon_seconds": 86400, "average_sentiment": 0.5, "trend": 0.6},
    ]
    feats = extract_logistic_features(horizons, lsr_signal=0.7, funding_signal=-0.8)
    assert feats == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, -0.8]
    assert len(feats) == N_FEATURES


def test_feature_vector_handles_missing():
    feats = extract_logistic_features([{"horizon_seconds": 3600, "average_sentiment": None, "trend": None}], None, None)
    assert feats == [0.0] * N_FEATURES


def test_logistic_model_score_bounds_and_direction():
    # Weight only on feature 0 (positive) -> higher feature 0 => higher score.
    m = LogisticModel(weights=[0.0, 5.0] + [0.0] * (N_FEATURES - 1),
                      means=[0.0] * N_FEATURES, stds=[1.0] * N_FEATURES, horizon=21600)
    hi = m.score([1.0] + [0.0] * (N_FEATURES - 1))
    lo = m.score([-1.0] + [0.0] * (N_FEATURES - 1))
    assert -1.0 <= lo < 0 < hi <= 1.0


def _samples(n, horizon, label_fn):
    out = []
    for i in range(n):
        a = ((i % 11) - 5) / 5.0
        feats = [a, 0.0, a, 0.0, a, 0.0, 0.0, 0.0]
        ret = label_fn(a)
        out.append((i * horizon, feats, 1 if ret > 0 else 0, ret))
    return out


def test_train_logistic_adopts_learnable_signal():
    # Returns driven by feature 0 with a sign the baseline (0.7*avg+0.3*trend) also
    # gets right -> learner must at least match; we use a relationship the baseline
    # gets WRONG so logistic clearly wins: ret = -a (contrarian on the 6h avg).
    s = _samples(360, 21600, lambda a: -a)
    rep = train_logistic(s, 21600)
    assert rep["adopted"] is True
    assert rep["model"] is not None


def test_train_logistic_refuses_too_few_independent():
    s = _samples(80, 21600, lambda a: -a)  # 80 samples, fewer than 80 independent OOS
    rep = train_logistic(s, 21600)
    assert rep["adopted"] is False


def test_model_json_round_trip_includes_logistic():
    m = CalibrationModel(trained_at=1.0, adopted=True,
                         logistic=LogisticModel(weights=[0.1] * (N_FEATURES + 1),
                                                means=[0.0] * N_FEATURES, stds=[1.0] * N_FEATURES, horizon=21600))
    back = CalibrationModel.from_json(m.to_json())
    assert back.logistic is not None
    assert back.logistic.horizon == 21600


def test_composite_uses_logistic_when_present():
    horizons = [
        {"horizon_seconds": 3600, "average_sentiment": 0.0, "trend": 0.0, "score": 0.0, "confidence": 1.0},
        {"horizon_seconds": 21600, "average_sentiment": 0.0, "trend": 0.0, "score": 0.0, "confidence": 1.0},
        {"horizon_seconds": 86400, "average_sentiment": 0.0, "trend": 0.0, "score": 0.0, "confidence": 1.0},
    ]
    # logistic keyed on the LSR feature (index 6), strong positive weight
    weights = [0.0] * (N_FEATURES + 1)
    weights[7] = 6.0  # bias idx0, so feature 6 -> weight idx 7
    model = CalibrationModel(trained_at=0, adopted=True, horizon_weights={3600: 1.0, 21600: 1.0, 86400: 1.0},
                             logistic=LogisticModel(weights=weights, means=[0.0] * N_FEATURES,
                                                    stds=[1.0] * N_FEATURES, horizon=21600))
    score, _ = composite_outlook(horizons, tilt=0.0, model=model, lsr_signal=1.0, funding_signal=0.0)
    assert score > 0.5  # logistic drove it positive via the LSR feature

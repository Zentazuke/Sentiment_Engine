"""Calibration learner: linear algebra, walk-forward adopt gate, integration."""

import json

from sentiment_engine.signals import calibration as cal
from sentiment_engine.signals.calibration import (
    CalibrationModel,
    HorizonModel,
    PriceSeries,
    Sample,
    build_samples,
    horizon_features,
    ridge_toward_prior,
    solve_linear,
    train_horizon,
)
from sentiment_engine.signals.outlook import composite_outlook, horizon_aggregate


# ---- linear algebra ----

def test_solve_linear_known_system():
    # 2x + y = 5 ; x + 3y = 10  ->  x=1, y=3
    sol = solve_linear([[2, 1], [1, 3]], [5, 10])
    assert abs(sol[0] - 1) < 1e-9 and abs(sol[1] - 3) < 1e-9


def test_ridge_recovers_planted_weights():
    # y = 2*a - 1*b, lots of clean data, weak prior pull -> recover ~[2,-1,0]
    rows = []
    for i in range(200):
        a = ((i % 7) - 3) / 3.0
        b = ((i % 5) - 2) / 2.0
        rows.append(([a, b, 1.0], 2 * a - 1 * b))
    w = ridge_toward_prior(rows, prior=[0.7, 0.3, 0.0], lam=0.01)
    assert abs(w[0] - 2) < 0.1 and abs(w[1] + 1) < 0.1 and abs(w[2]) < 0.1


def test_ridge_stays_near_prior_with_no_data():
    w = ridge_toward_prior([], prior=[0.7, 0.3, 0.0], lam=5.0)
    assert w == [0.7, 0.3, 0.0]


# ---- feature / outcome extraction ----

def test_horizon_features_parses():
    hj = json.dumps([{"horizon_seconds": 3600, "average_sentiment": 0.4, "trend": -0.1}])
    assert horizon_features(hj, 3600) == (0.4, -0.1)
    assert horizon_features(hj, 21600) is None  # horizon absent


def test_horizon_features_none_when_no_sentiment():
    hj = json.dumps([{"horizon_seconds": 3600, "average_sentiment": None, "trend": 0.0}])
    assert horizon_features(hj, 3600) is None


def test_build_samples_computes_forward_return():
    prices = PriceSeries([(0.0, 100.0), (3600.0, 110.0)])
    hj = json.dumps([{"horizon_seconds": 3600, "average_sentiment": 0.5, "trend": 0.0}])
    samples = build_samples([(0.0, hj, 100.0)], prices, 3600)
    assert len(samples) == 1
    assert abs(samples[0].ret - 0.10) < 1e-9  # 100 -> 110


# ---- walk-forward adopt gate ----

def _samples(n, horizon, ret_fn):
    return [Sample(t=i * horizon, avg=((i % 9) - 4) / 4.0, trend=0.0, ret=ret_fn(((i % 9) - 4) / 4.0, i))
            for i in range(n)]


def test_adopts_when_learned_beats_default():
    # Returns are ANTI-correlated with sentiment: default (positive w_avg) is
    # wrong, the learner should flip the sign and win out-of-sample.
    s = _samples(240, 3600, lambda avg, i: -0.4 * avg)
    rep = train_horizon(s, 3600)
    assert rep["adopted"] is True
    assert rep["model"]["w_avg"] < 0  # learned the contrarian relationship


def test_does_not_adopt_on_noise():
    # Returns independent of sentiment (deterministic alternating) -> ridge keeps
    # weights ~prior, learned ~= default, gate refuses.
    s = _samples(240, 3600, lambda avg, i: 0.01 if i % 2 else -0.01)
    rep = train_horizon(s, 3600)
    assert rep["adopted"] is False


def test_does_not_adopt_with_too_few_samples():
    s = _samples(20, 3600, lambda avg, i: -0.4 * avg)
    rep = train_horizon(s, 3600)
    assert rep["adopted"] is False
    assert "too few" in rep["reason"]


# ---- model serialization ----

def test_model_json_round_trip():
    m = CalibrationModel(trained_at=123.0, adopted=True,
                         horizons={3600: HorizonModel(-0.5, 0.2, 0.01, 0.02)})
    back = CalibrationModel.from_json(m.to_json())
    assert back.adopted is True
    assert abs(back.horizons[3600].w_avg + 0.5) < 1e-9


def test_horizon_model_score_bounds_and_sign():
    hm = HorizonModel(w_avg=-1.0, w_trend=0.0, w_bias=0.0, return_scale=0.05)
    assert -1.0 <= hm.score(0.8, 0.0) <= 1.0
    assert hm.score(0.8, 0.0) < 0  # negative weight flips a positive avg


# ---- scorer integration ----

def test_horizon_aggregate_uses_learned_head():
    rows = [(100.0, "news:x", 0.6), (150.0, "news:x", 0.6)]
    model = CalibrationModel(trained_at=0, adopted=True,
                             horizons={3600: HorizonModel(-1.0, 0.0, 0.0, 0.05)})
    default = horizon_aggregate(rows, now=200.0, horizon_seconds=3600, model=None)
    learned = horizon_aggregate(rows, now=200.0, horizon_seconds=3600, model=model)
    assert default["score"] > 0          # positive sentiment -> positive default
    assert learned["score"] < 0          # learned contrarian head flips it
    assert learned["calibrated"] is True


def test_composite_uses_model_weights():
    horizons = [
        {"horizon_seconds": 3600, "score": 1.0, "confidence": 1.0},
        {"horizon_seconds": 21600, "score": -1.0, "confidence": 1.0},
    ]
    model = CalibrationModel(trained_at=0, adopted=True,
                             horizon_weights={3600: 1.0, 21600: 0.0}, tilt_coef=0.0)
    score, _ = composite_outlook(horizons, tilt=0.5, model=model)
    assert score == 1.0  # all weight on 3600 (score 1.0), tilt zeroed by tilt_coef

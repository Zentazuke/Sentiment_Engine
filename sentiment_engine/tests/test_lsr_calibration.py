"""Learner side of crowd positioning: adopt gate + composite application."""

from sentiment_engine.signals.calibration import (
    CalibrationModel,
    PriceSeries,
    Sample,
    build_lsr_samples,
    train,
    train_lsr,
)
from sentiment_engine.signals.outlook import composite_outlook
import json


def _lsr_samples(n, horizon, ret_fn):
    out = []
    for i in range(n):
        lsr = ((i % 9) - 4) / 4.0  # spread over [-1, 1]
        out.append(Sample(t=i * horizon, avg=lsr, trend=0.0, ret=ret_fn(lsr)))
    return out


def test_lsr_adopts_contrarian_signal():
    # Crowd net-long precedes DOWN moves -> contrarian; learner should adopt a
    # negative coefficient.
    s = _lsr_samples(240, 21600, lambda lsr: -0.3 * lsr)
    rep = train_lsr(s, 21600)
    assert rep["adopted"] is True
    assert rep["lsr_coef"] < 0
    assert rep["direction"] == "contrarian"


def test_lsr_does_not_adopt_noise():
    s = _lsr_samples(240, 21600, lambda lsr: 0.01)  # constant, no relationship
    rep = train_lsr(s, 21600)
    assert rep["adopted"] is False


def test_build_lsr_samples_joins_positioning_and_returns():
    positioning = PriceSeries([(0.0, 0.5)])         # signal at outlook time
    prices = PriceSeries([(0.0, 100.0), (21600.0, 90.0)])  # -10% over horizon
    outlooks = [(0.0, "{}", 100.0)]
    samples = build_lsr_samples(outlooks, positioning, prices, 21600)
    assert len(samples) == 1
    assert samples[0].avg == 0.5
    assert abs(samples[0].ret + 0.10) < 1e-9


def test_composite_applies_lsr_coef():
    horizons = [{"horizon_seconds": 3600, "score": 0.0, "confidence": 1.0}]
    model = CalibrationModel(trained_at=0, adopted=True,
                             horizon_weights={3600: 1.0}, tilt_coef=0.0, lsr_coef=-0.3)
    # crowd net long (+1) with a contrarian coef (-0.3) -> bearish nudge
    score, _ = composite_outlook(horizons, tilt=0.0, model=model, lsr_signal=1.0)
    assert abs(score + 0.3) < 1e-9


def test_lsr_coef_survives_json_round_trip():
    m = CalibrationModel(trained_at=1.0, adopted=True, lsr_coef=-0.21)
    assert abs(CalibrationModel.from_json(m.to_json()).lsr_coef + 0.21) < 1e-9


def test_train_end_to_end_sets_lsr_coef():
    # Build synthetic outlooks + prices + positioning where crowd-long -> down.
    horizon = 21600
    outlooks, price_pts, pos_pts = [], [], []
    for i in range(260):
        t = i * horizon
        lsr = ((i % 9) - 4) / 4.0
        p0 = 100.0
        p1 = 100.0 * (1 - 0.3 * lsr * 0.05)  # contrarian: long crowd -> lower
        outlooks.append((t, json.dumps([{"horizon_seconds": horizon, "average_sentiment": None}]), p0))
        price_pts.append((t, p0))
        price_pts.append((t + horizon, p1))
        pos_pts.append((t, lsr))
    model = train(
        {"BTC/USDT": outlooks},
        {"BTC/USDT": PriceSeries(price_pts)},
        positioning_by_symbol={"BTC/USDT": PriceSeries(pos_pts)},
    )
    assert model.report["lsr"]["adopted"] is True
    assert model.lsr_coef < 0
    assert model.adopted is True

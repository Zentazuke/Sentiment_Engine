"""Learner side of the funding-rate signal: adopt gate + composite application."""

import json

from sentiment_engine.signals.calibration import CalibrationModel, PriceSeries, Sample, train, train_lsr
from sentiment_engine.signals.outlook import composite_outlook


def _samples(n, horizon, ret_fn):
    return [Sample(t=i * horizon, avg=((i % 9) - 4) / 4.0, trend=0.0, ret=ret_fn(((i % 9) - 4) / 4.0))
            for i in range(n)]


def test_funding_coef_survives_json_round_trip():
    m = CalibrationModel(trained_at=1.0, adopted=True, funding_coef=-0.18)
    assert abs(CalibrationModel.from_json(m.to_json()).funding_coef + 0.18) < 1e-9


def test_composite_applies_funding_coef():
    horizons = [{"horizon_seconds": 3600, "score": 0.0, "confidence": 1.0}]
    model = CalibrationModel(trained_at=0, adopted=True,
                             horizon_weights={3600: 1.0}, tilt_coef=0.0, lsr_coef=0.0, funding_coef=0.25)
    # positive funding crowding (+1) with a momentum coef (+0.25) -> bullish nudge
    score, _ = composite_outlook(horizons, tilt=0.0, model=model, funding_signal=1.0)
    assert abs(score - 0.25) < 1e-9


def test_composite_combines_lsr_and_funding():
    horizons = [{"horizon_seconds": 3600, "score": 0.0, "confidence": 1.0}]
    model = CalibrationModel(trained_at=0, adopted=True, horizon_weights={3600: 1.0},
                             tilt_coef=0.0, lsr_coef=-0.2, funding_coef=0.1)
    score, _ = composite_outlook(horizons, tilt=0.0, model=model, lsr_signal=1.0, funding_signal=1.0)
    assert abs(score - (-0.2 + 0.1)) < 1e-9


def test_train_end_to_end_sets_funding_coef():
    horizon = 21600
    outlooks, price_pts, fund_pts = [], [], []
    for i in range(260):
        t = i * horizon
        f = ((i % 9) - 4) / 4.0
        p0 = 100.0
        p1 = 100.0 * (1 + 0.3 * f * 0.05)  # momentum: positive funding -> higher
        outlooks.append((t, json.dumps([{"horizon_seconds": horizon, "average_sentiment": None}]), p0))
        price_pts.append((t, p0))
        price_pts.append((t + horizon, p1))
        fund_pts.append((t, f))
    model = train(
        {"BTC/USDT": outlooks},
        {"BTC/USDT": PriceSeries(price_pts)},
        funding_by_symbol={"BTC/USDT": PriceSeries(fund_pts)},
    )
    assert model.report["funding"]["adopted"] is True
    assert model.funding_coef > 0  # learned momentum direction
    assert model.adopted is True


def test_no_funding_data_keeps_default():
    model = train({"BTC/USDT": []}, {"BTC/USDT": PriceSeries([])})
    assert model.funding_coef == 0.0
    assert model.report["funding"]["adopted"] is False

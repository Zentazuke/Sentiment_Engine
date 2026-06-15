"""Walk-forward outlook calibration (the engine's learning loop).

The outlook scorer ships with hand-set coefficients (0.7*avg_sentiment +
0.3*trend per horizon, fixed horizon weights, a context tilt). This module
*learns* those coefficients from journaled outcomes: for each matured outlook
it knows the realized forward return, so it can refit the mapping from
sentiment features to realized direction.

Design principles (this feeds a trading-adjacent tool, so guardrails dominate):

1. Interpretable, not a black box. We learn a small linear model per horizon
   (w_avg, w_trend, w_bias) plus composite horizon weights and a tilt
   coefficient. Every number is inspectable.
2. Ridge regularization TOWARD the current defaults, so thin data barely moves
   the coefficients and the model degrades gracefully to today's behaviour.
3. Walk-forward, out-of-sample evaluation. Train on the earlier slice, score on
   the later slice -- never on data the model has seen.
4. De-overlap the evaluation. Outlooks are written every ~minute but horizons
   are hours, so raw samples are massively autocorrelated; OOS skill is measured
   on stride-spaced, near-independent samples.
5. Adopt-only-if-better. The learned model replaces the default ONLY if it beats
   the default on the out-of-sample, de-overlapped set by a margin AND there are
   enough independent samples. Otherwise we keep the default and report why.

Pure-Python (no numpy) so it stays dependency-free and unit-testable offline.
"""

from __future__ import annotations

import bisect
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from sentiment_engine.signals.logistic_model import (
    LOGISTIC_HORIZON,
    LogisticModel,
    extract_logistic_features,
    train_logistic,
)

# --- defaults / prior (must mirror signals/outlook.py) ---
PRIOR_W_AVG = 0.7
PRIOR_W_TREND = 0.3
PRIOR_W_BIAS = 0.0
DEFAULT_HORIZONS = (3600, 21600, 86400)
DEFAULT_HORIZON_WEIGHTS = {3600: 0.5, 21600: 0.3, 86400: 0.2}
DEFAULT_TILT_COEF = 1.0

# --- adopt-gate thresholds ---
MIN_INDEPENDENT_SAMPLES = 40   # per horizon, on the OOS de-overlapped set
ADOPT_ACC_MARGIN = 0.03        # learned must beat default OOS accuracy by >= this
RIDGE_LAMBDA = 5.0             # pull toward prior; higher = more conservative
LSR_COEF_CAP = 0.30            # max absolute crowd-positioning shift on the composite
LSR_HORIZON = 21600            # horizon (6h) used to learn the positioning coefficient


# ----------------------------- small linear algebra -----------------------------

def solve_linear(matrix: List[List[float]], rhs: List[float]) -> Optional[List[float]]:
    """Solve A x = b for small A via Gaussian elimination with partial pivoting."""
    n = len(matrix)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        piv = a[col][col]
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col] / piv
            for c in range(col, n + 1):
                a[r][c] -= factor * a[col][c]
    return [a[i][n] / a[i][i] for i in range(n)]


def ridge_toward_prior(
    rows: Sequence[Tuple[List[float], float]], prior: List[float], lam: float
) -> List[float]:
    """Ridge regression whose penalty pulls weights toward `prior` instead of 0.

    Minimizes sum((y - x.w)^2) + lam * ||w - prior||^2.
    Closed form: w = (X^T X + lam I)^-1 (X^T y + lam * prior).
    """
    dim = len(prior)
    ata = [[0.0] * dim for _ in range(dim)]
    aty = [0.0] * dim
    for x, y in rows:
        for i in range(dim):
            aty[i] += x[i] * y
            for j in range(dim):
                ata[i][j] += x[i] * x[j]
    for i in range(dim):
        ata[i][i] += lam
        aty[i] += lam * prior[i]
    solved = solve_linear(ata, aty)
    return solved if solved is not None else list(prior)


# ----------------------------- feature / outcome extraction -----------------------------

def horizon_features(horizons_json: str, horizon_seconds: int) -> Optional[Tuple[float, float]]:
    """(avg_sentiment, trend) for one horizon, or None if not scorable."""
    try:
        horizons = json.loads(horizons_json)
    except (TypeError, ValueError):
        return None
    for h in horizons:
        if int(h.get("horizon_seconds", -1)) == horizon_seconds:
            avg = h.get("average_sentiment")
            if avg is None:
                return None
            return float(avg), float(h.get("trend") or 0.0)
    return None


class PriceSeries:
    """Time-sorted price lookup with nearest-within-tolerance semantics."""

    def __init__(self, points: Sequence[Tuple[float, float]]):
        pts = sorted((t, p) for t, p in points if p is not None)
        self._t = [t for t, _ in pts]
        self._p = [p for _, p in pts]

    def at(self, ts: float, tol: float = 180.0) -> Optional[float]:
        if not self._t:
            return None
        i = bisect.bisect_left(self._t, ts)
        best, best_d = None, tol + 1
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(self._t):
                d = abs(self._t[j] - ts)
                if d < best_d:
                    best, best_d = self._p[j], d
        return best if best_d <= tol else None


@dataclass
class Sample:
    t: float
    avg: float
    trend: float
    ret: float  # realized forward return over the horizon


def build_samples(
    outlooks: Sequence[Tuple[float, str, float]],  # (computed_at, horizons_json, price_at_compute)
    prices: PriceSeries,
    horizon_seconds: int,
) -> List[Sample]:
    """Join matured outlooks to realized forward returns for one horizon."""
    samples: List[Sample] = []
    for computed_at, horizons_json, price_at_compute in outlooks:
        feats = horizon_features(horizons_json, horizon_seconds)
        if feats is None:
            continue
        p0 = price_at_compute if price_at_compute else prices.at(computed_at)
        p1 = prices.at(computed_at + horizon_seconds)
        if not p0 or not p1:
            continue
        samples.append(Sample(computed_at, feats[0], feats[1], (p1 - p0) / p0))
    samples.sort(key=lambda s: s.t)
    return samples


# ----------------------------- evaluation helpers -----------------------------

def _deoverlap(samples: Sequence[Sample], horizon_seconds: int) -> List[Sample]:
    """Keep samples spaced >= horizon apart so accuracy isn't autocorrelation."""
    kept: List[Sample] = []
    last_t = -math.inf
    for s in samples:
        if s.t - last_t >= horizon_seconds:
            kept.append(s)
            last_t = s.t
    return kept


def _directional_accuracy(samples: Sequence[Sample], w_avg: float, w_trend: float, w_bias: float) -> Optional[float]:
    hits = total = 0
    for s in samples:
        pred = w_avg * s.avg + w_trend * s.trend + w_bias
        if abs(pred) < 1e-9 or abs(s.ret) < 1e-12:
            continue
        total += 1
        if (pred > 0) == (s.ret > 0):
            hits += 1
    return hits / total if total else None


# ----------------------------- model -----------------------------

@dataclass
class HorizonModel:
    w_avg: float
    w_trend: float
    w_bias: float
    return_scale: float  # std of training returns; maps raw pred -> [-1,1] via tanh

    def score(self, avg: float, trend: float) -> float:
        raw = self.w_avg * avg + self.w_trend * trend + self.w_bias
        scale = self.return_scale or 1e-6
        return max(-1.0, min(1.0, math.tanh(raw / scale)))


@dataclass
class CalibrationModel:
    trained_at: float
    adopted: bool
    horizons: Dict[int, HorizonModel] = field(default_factory=dict)
    horizon_weights: Dict[int, float] = field(default_factory=lambda: dict(DEFAULT_HORIZON_WEIGHTS))
    tilt_coef: float = DEFAULT_TILT_COEF
    lsr_coef: float = 0.0  # crowd-positioning coefficient (0 until the learner adopts one)
    funding_coef: float = 0.0  # funding-rate coefficient (0 until the learner adopts one)
    logistic: Optional[LogisticModel] = None  # multi-feature model; supersedes the blend when adopted
    report: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "version": 1,
            "trained_at": self.trained_at,
            "adopted": self.adopted,
            "horizons": {str(k): asdict(v) for k, v in self.horizons.items()},
            "horizon_weights": {str(k): v for k, v in self.horizon_weights.items()},
            "tilt_coef": self.tilt_coef,
            "lsr_coef": self.lsr_coef,
            "funding_coef": self.funding_coef,
            "logistic": self.logistic.to_dict() if self.logistic else None,
            "report": self.report,
        }, indent=2)

    @staticmethod
    def from_json(text: str) -> "CalibrationModel":
        d = json.loads(text)
        return CalibrationModel(
            trained_at=d.get("trained_at", 0.0),
            adopted=bool(d.get("adopted", False)),
            horizons={int(k): HorizonModel(**v) for k, v in d.get("horizons", {}).items()},
            horizon_weights={int(k): float(v) for k, v in d.get("horizon_weights", {}).items()}
                            or dict(DEFAULT_HORIZON_WEIGHTS),
            tilt_coef=float(d.get("tilt_coef", DEFAULT_TILT_COEF)),
            lsr_coef=float(d.get("lsr_coef", 0.0)),
            funding_coef=float(d.get("funding_coef", 0.0)),
            logistic=LogisticModel.from_dict(d["logistic"]) if d.get("logistic") else None,
            report=d.get("report", {}),
        )


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def train_horizon(samples: Sequence[Sample], horizon_seconds: int) -> dict:
    """Walk-forward fit + adopt decision for one horizon. Returns a report dict."""
    prior = [PRIOR_W_AVG, PRIOR_W_TREND, PRIOR_W_BIAS]
    n = len(samples)
    report = {
        "horizon_seconds": horizon_seconds, "n_samples": n,
        "adopted": False, "reason": "", "model": None,
    }
    if n < 30:
        report["reason"] = f"too few samples ({n} < 30)"
        return report

    split = int(n * 0.7)
    train, test = list(samples[:split]), list(samples[split:])
    rows = [([s.avg, s.trend, 1.0], s.ret) for s in train]
    w = ridge_toward_prior(rows, prior, RIDGE_LAMBDA)

    test_indep = _deoverlap(test, horizon_seconds)
    n_indep = len(test_indep)
    learned_acc = _directional_accuracy(test_indep, w[0], w[1], w[2])
    default_acc = _directional_accuracy(test_indep, PRIOR_W_AVG, PRIOR_W_TREND, PRIOR_W_BIAS)

    report.update({
        "n_independent_oos": n_indep,
        "learned_weights": {"w_avg": round(w[0], 4), "w_trend": round(w[1], 4), "w_bias": round(w[2], 4)},
        "oos_accuracy_learned": round(learned_acc, 4) if learned_acc is not None else None,
        "oos_accuracy_default": round(default_acc, 4) if default_acc is not None else None,
    })

    if n_indep < MIN_INDEPENDENT_SAMPLES:
        report["reason"] = f"too few independent OOS samples ({n_indep} < {MIN_INDEPENDENT_SAMPLES})"
        return report
    if learned_acc is None or default_acc is None:
        report["reason"] = "insufficient directional samples"
        return report
    if learned_acc < default_acc + ADOPT_ACC_MARGIN:
        report["reason"] = (f"learned {learned_acc:.3f} did not beat default {default_acc:.3f} "
                            f"by {ADOPT_ACC_MARGIN}")
        return report

    return_scale = _std([s.ret for s in train]) or 1e-6
    report["adopted"] = True
    report["reason"] = f"learned {learned_acc:.3f} beats default {default_acc:.3f}"
    report["model"] = {"w_avg": w[0], "w_trend": w[1], "w_bias": w[2], "return_scale": return_scale}
    return report


def build_lsr_samples(
    outlooks: Sequence[Tuple[float, str, float]],
    positioning: PriceSeries,
    prices: PriceSeries,
    horizon_seconds: int,
) -> List[Sample]:
    """Join outlooks to (crowd positioning signal at compute time, forward return)."""
    samples: List[Sample] = []
    for computed_at, _hj, price_at_compute in outlooks:
        lsr = positioning.at(computed_at, tol=600.0)  # positioning updates ~5 min
        if lsr is None:
            continue
        p0 = price_at_compute if price_at_compute else prices.at(computed_at)
        p1 = prices.at(computed_at + horizon_seconds)
        if not p0 or not p1:
            continue
        samples.append(Sample(computed_at, lsr, 0.0, (p1 - p0) / p0))
    samples.sort(key=lambda s: s.t)
    return samples


def train_lsr(samples: Sequence[Sample], horizon_seconds: int) -> dict:
    """Learn the crowd-positioning coefficient (sign = momentum vs contrarian).

    Single through-origin fit ret ~ c * lsr_signal, walk-forward validated on
    de-overlapped out-of-sample data. Baseline is no-skill (0.5); adopts only if
    the learned sign beats it by the margin with enough independent samples.
    Magnitude scales with out-of-sample skill, capped at LSR_COEF_CAP.
    """
    report = {"n_samples": len(samples), "adopted": False, "reason": "", "lsr_coef": 0.0}
    n = len(samples)
    if n < 30:
        report["reason"] = f"too few samples ({n} < 30)"
        return report
    split = int(n * 0.7)
    train_s, test_s = list(samples[:split]), list(samples[split:])
    denom = sum(s.avg * s.avg for s in train_s)
    if denom < 1e-12:
        report["reason"] = "no positioning variance in training set"
        return report
    c = sum(s.avg * s.ret for s in train_s) / denom

    test_indep = _deoverlap(test_s, horizon_seconds)
    n_indep = len(test_indep)
    oos_acc = _directional_accuracy(test_indep, c, 0.0, 0.0)
    report.update({
        "n_independent_oos": n_indep,
        "oos_accuracy": round(oos_acc, 4) if oos_acc is not None else None,
        "direction": "contrarian" if c < 0 else "momentum",
    })
    if n_indep < MIN_INDEPENDENT_SAMPLES:
        report["reason"] = f"too few independent OOS samples ({n_indep} < {MIN_INDEPENDENT_SAMPLES})"
        return report
    if oos_acc is None or oos_acc < 0.5 + ADOPT_ACC_MARGIN:
        report["reason"] = f"OOS accuracy {oos_acc} did not beat 0.5 by {ADOPT_ACC_MARGIN}"
        return report
    skill = max(0.0, min(1.0, (oos_acc - 0.5) / 0.5))
    coef = (1.0 if c > 0 else -1.0) * LSR_COEF_CAP * skill
    report["adopted"] = True
    report["reason"] = f"OOS {oos_acc:.3f} beats 0.5 ({report['direction']})"
    report["lsr_coef"] = round(coef, 4)
    return report


def build_logistic_samples(outlooks, prices, positioning, funding, horizon_seconds):
    """Per outlook: full feature vector + forward-return direction at one horizon."""
    samples = []
    for computed_at, horizons_json, price_at_compute in outlooks:
        try:
            hs = json.loads(horizons_json)
        except (TypeError, ValueError):
            continue
        lsr = positioning.at(computed_at, tol=600.0) if positioning is not None else None
        fund = funding.at(computed_at, tol=600.0) if funding is not None else None
        feats = extract_logistic_features(hs, lsr, fund)
        p0 = price_at_compute if price_at_compute else prices.at(computed_at)
        p1 = prices.at(computed_at + horizon_seconds)
        if not p0 or not p1:
            continue
        ret = (p1 - p0) / p0
        samples.append((computed_at, feats, 1 if ret > 0 else 0, ret))
    samples.sort(key=lambda s: s[0])
    return samples


def train(
    outlooks_by_symbol: Dict[str, Sequence[Tuple[float, str, float]]],
    prices_by_symbol: Dict[str, PriceSeries],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    positioning_by_symbol: Optional[Dict[str, PriceSeries]] = None,
    funding_by_symbol: Optional[Dict[str, PriceSeries]] = None,
) -> CalibrationModel:
    """Train across all symbols pooled (mood->return mapping is cross-asset).

    Returns a CalibrationModel; `adopted` is True only if at least one horizon
    cleared the gate. Non-adopted horizons keep the default coefficients.
    """
    model = CalibrationModel(trained_at=time.time(), adopted=False)
    per_horizon_reports = {}
    any_adopted = False
    for h in horizons:
        pooled: List[Sample] = []
        for sym, outlooks in outlooks_by_symbol.items():
            prices = prices_by_symbol.get(sym)
            if prices is None:
                continue
            pooled.extend(build_samples(outlooks, prices, h))
        pooled.sort(key=lambda s: s.t)
        rep = train_horizon(pooled, h)
        per_horizon_reports[str(h)] = rep
        if rep["adopted"] and rep["model"]:
            m = rep["model"]
            model.horizons[h] = HorizonModel(m["w_avg"], m["w_trend"], m["w_bias"], m["return_scale"])
            any_adopted = True

    lsr_report = {"adopted": False, "reason": "no positioning data supplied"}
    if positioning_by_symbol:
        pooled_lsr: List[Sample] = []
        for sym, outlooks in outlooks_by_symbol.items():
            prices = prices_by_symbol.get(sym)
            positioning = positioning_by_symbol.get(sym)
            if prices is None or positioning is None:
                continue
            pooled_lsr.extend(build_lsr_samples(outlooks, positioning, prices, LSR_HORIZON))
        pooled_lsr.sort(key=lambda s: s.t)
        lsr_report = train_lsr(pooled_lsr, LSR_HORIZON)
        if lsr_report["adopted"]:
            model.lsr_coef = lsr_report["lsr_coef"]
            any_adopted = True

    funding_report = {"adopted": False, "reason": "no funding data supplied"}
    if funding_by_symbol:
        pooled_fund: List[Sample] = []
        for sym, outlooks in outlooks_by_symbol.items():
            prices = prices_by_symbol.get(sym)
            funding = funding_by_symbol.get(sym)
            if prices is None or funding is None:
                continue
            pooled_fund.extend(build_lsr_samples(outlooks, funding, prices, LSR_HORIZON))
        pooled_fund.sort(key=lambda s: s.t)
        funding_report = train_lsr(pooled_fund, LSR_HORIZON)
        funding_report["funding_coef"] = funding_report.pop("lsr_coef", 0.0)
        if funding_report["adopted"]:
            model.funding_coef = funding_report["funding_coef"]
            any_adopted = True

    pooled_log: List = []
    for sym, outlooks in outlooks_by_symbol.items():
        prices = prices_by_symbol.get(sym)
        if prices is None:
            continue
        pos = (positioning_by_symbol or {}).get(sym)
        fund = (funding_by_symbol or {}).get(sym)
        pooled_log.extend(build_logistic_samples(outlooks, prices, pos, fund, LOGISTIC_HORIZON))
    pooled_log.sort(key=lambda s: s[0])
    logistic_report = train_logistic(pooled_log, LOGISTIC_HORIZON)
    if logistic_report["adopted"] and logistic_report["model"]:
        model.logistic = LogisticModel.from_dict(logistic_report["model"])
        any_adopted = True

    model.adopted = any_adopted
    model.report = {
        "trained_at": model.trained_at, "horizons": per_horizon_reports,
        "lsr": lsr_report, "funding": funding_report, "logistic": logistic_report,
    }
    return model

"""Multi-feature logistic calibration model (pure Python, no dependencies).

The base calibrator fits each horizon separately and adds the positioning /
funding signals as scalar coefficients. This model instead pools EVERY feature
into one L2-regularized logistic regression that predicts the direction of the
forward return:

    features = [avg_1h, trend_1h, avg_6h, trend_6h, avg_24h, trend_24h, lsr, funding]

It is held to a stricter bar than the scalar coefficients (more features need
more data): it is adopted only if it beats the hand-set sentiment baseline on a
walk-forward, de-overlapped out-of-sample set AND there are enough independent
samples. Until then the engine keeps its existing behaviour. Everything here is
plain Python and unit-testable offline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_HORIZONS = (3600, 21600, 86400)
LOGISTIC_HORIZON = 21600            # 6h: the direction the model predicts
N_FEATURES = 8                     # 3 horizons x (avg, trend) + lsr + funding
MIN_LOGISTIC_OOS = 80              # independent OOS samples required to adopt
ADOPT_MARGIN = 0.03                # must beat the sentiment baseline by this
L2 = 2.0
ITERS = 400
LR = 0.3
# Index of the 6h avg / trend features, used for the sentiment baseline.
_BASE_AVG_IDX = DEFAULT_HORIZONS.index(LOGISTIC_HORIZON) * 2
_BASE_TREND_IDX = _BASE_AVG_IDX + 1


def extract_logistic_features(
    horizons: Sequence[dict], lsr_signal: Optional[float], funding_signal: Optional[float]
) -> List[float]:
    """Build the fixed-length feature vector from horizon rows + market signals."""
    hmap = {int(h.get("horizon_seconds", -1)): h for h in horizons}
    feats: List[float] = []
    for hs in DEFAULT_HORIZONS:
        h = hmap.get(hs)
        avg = h.get("average_sentiment") if h else None
        trend = h.get("trend") if h else None
        feats.append(float(avg) if avg is not None else 0.0)
        feats.append(float(trend) if trend is not None else 0.0)
    feats.append(float(lsr_signal) if lsr_signal is not None else 0.0)
    feats.append(float(funding_signal) if funding_signal is not None else 0.0)
    return feats


def _sigmoid(z: float) -> float:
    if z < -35:
        return 0.0
    if z > 35:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class LogisticModel:
    weights: List[float]   # length N_FEATURES + 1 (index 0 = bias)
    means: List[float]     # length N_FEATURES
    stds: List[float]      # length N_FEATURES
    horizon: int

    def _standardize(self, feats: Sequence[float]) -> List[float]:
        return [(feats[i] - self.means[i]) / self.stds[i] for i in range(N_FEATURES)]

    def probability(self, feats: Sequence[float]) -> float:
        z = self.weights[0]
        std = self._standardize(feats)
        for i in range(N_FEATURES):
            z += self.weights[i + 1] * std[i]
        return _sigmoid(z)

    def score(self, feats: Sequence[float]) -> float:
        """Map P(up) -> outlook score in [-1, 1]."""
        return max(-1.0, min(1.0, 2.0 * self.probability(feats) - 1.0))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "LogisticModel":
        return LogisticModel(
            weights=[float(x) for x in d["weights"]],
            means=[float(x) for x in d["means"]],
            stds=[float(x) for x in d["stds"]],
            horizon=int(d["horizon"]),
        )


def _standardize_params(rows: Sequence[Tuple[List[float], int]]) -> Tuple[List[float], List[float]]:
    n = len(rows)
    means = [0.0] * N_FEATURES
    for feats, _ in rows:
        for i in range(N_FEATURES):
            means[i] += feats[i]
    means = [m / n for m in means]
    stds = [0.0] * N_FEATURES
    for feats, _ in rows:
        for i in range(N_FEATURES):
            stds[i] += (feats[i] - means[i]) ** 2
    stds = [math.sqrt(s / n) if s > 0 else 1.0 for s in stds]  # constant feature -> 1 (contributes 0)
    return means, stds


def fit_logistic(rows: Sequence[Tuple[List[float], int]], means: List[float], stds: List[float]) -> List[float]:
    """L2-regularized logistic regression via gradient descent. Returns weights."""
    w = [0.0] * (N_FEATURES + 1)
    n = len(rows)
    if n == 0:
        return w
    std_rows = [([(feats[i] - means[i]) / stds[i] for i in range(N_FEATURES)], y) for feats, y in rows]
    for _ in range(ITERS):
        grad = [0.0] * (N_FEATURES + 1)
        for feats, y in std_rows:
            z = w[0] + sum(w[i + 1] * feats[i] for i in range(N_FEATURES))
            err = _sigmoid(z) - y
            grad[0] += err
            for i in range(N_FEATURES):
                grad[i + 1] += err * feats[i]
        for i in range(1, N_FEATURES + 1):
            grad[i] += L2 * w[i]  # L2 on weights, not bias
        for i in range(N_FEATURES + 1):
            w[i] -= LR * grad[i] / n
    return w


# (timestamp, features, y, realized_return)
LogiSample = Tuple[float, List[float], int, float]


def _deoverlap(samples: Sequence[LogiSample], horizon: int) -> List[LogiSample]:
    kept: List[LogiSample] = []
    last = -math.inf
    for s in samples:
        if s[0] - last >= horizon:
            kept.append(s)
            last = s[0]
    return kept


def _accuracy(samples: Sequence[LogiSample], predict) -> Optional[float]:
    hits = total = 0
    for _t, feats, _y, ret in samples:
        pred = predict(feats)
        if abs(pred) < 1e-12 or abs(ret) < 1e-12:
            continue
        total += 1
        if (pred > 0) == (ret > 0):
            hits += 1
    return hits / total if total else None


def train_logistic(samples: Sequence[LogiSample], horizon: int) -> dict:
    """Walk-forward fit + adopt decision. Returns a report dict (model under 'model')."""
    report = {"n_samples": len(samples), "adopted": False, "reason": "", "model": None}
    n = len(samples)
    if n < 60:
        report["reason"] = f"too few samples ({n} < 60)"
        return report
    split = int(n * 0.7)
    train, test = list(samples[:split]), list(samples[split:])
    rows = [(s[1], s[2]) for s in train]
    means, stds = _standardize_params(rows)
    weights = fit_logistic(rows, means, stds)
    model = LogisticModel(weights=weights, means=means, stds=stds, horizon=horizon)

    test_indep = _deoverlap(test, horizon)
    n_indep = len(test_indep)
    logi_acc = _accuracy(test_indep, model.score)
    base_acc = _accuracy(test_indep, lambda f: 0.7 * f[_BASE_AVG_IDX] + 0.3 * f[_BASE_TREND_IDX])
    report.update({
        "n_independent_oos": n_indep,
        "oos_accuracy_logistic": round(logi_acc, 4) if logi_acc is not None else None,
        "oos_accuracy_baseline": round(base_acc, 4) if base_acc is not None else None,
    })
    if n_indep < MIN_LOGISTIC_OOS:
        report["reason"] = f"too few independent OOS samples ({n_indep} < {MIN_LOGISTIC_OOS})"
        return report
    if logi_acc is None or base_acc is None:
        report["reason"] = "insufficient directional samples"
        return report
    if logi_acc < base_acc + ADOPT_MARGIN:
        report["reason"] = f"logistic {logi_acc:.3f} did not beat baseline {base_acc:.3f} by {ADOPT_MARGIN}"
        return report
    report["adopted"] = True
    report["reason"] = f"logistic {logi_acc:.3f} beats baseline {base_acc:.3f}"
    report["model"] = model.to_dict()
    return report

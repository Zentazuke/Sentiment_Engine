"""Train the outlook calibration model from the journal, with guardrails.

    python -m sentiment_engine.signals.calibrate                 # uses live DB
    python -m sentiment_engine.signals.calibrate --db sentiment_journal_snapshot.db
    python -m sentiment_engine.signals.calibrate --dry-run       # report only, don't write

Reads outlook_history, joins each matured outlook to its realized forward
return, fits a ridge model per horizon, walk-forward validates it, and writes
calibration_model.json ONLY describing what it found. The live model is adopted
only for horizons that beat the default out-of-sample (see calibration.py).
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from typing import Dict, List, Tuple

from sentiment_engine.config import CALIBRATION_MODEL_PATH, JOURNAL_DB_PATH
from sentiment_engine.signals.calibration import PriceSeries, train


def _table_exists(conn, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def load_journal(db_path: str):
    from sentiment_engine.storage.positioning_history import (
        long_fraction_signal, ratio_to_long_fraction,
    )
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT symbol, computed_at, horizons_json, price_at_compute"
        " FROM outlook_history WHERE price_at_compute IS NOT NULL ORDER BY computed_at"
    ).fetchall()

    outlooks: Dict[str, List[Tuple[float, str, float]]] = defaultdict(list)
    price_points: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for symbol, computed_at, horizons_json, price in rows:
        outlooks[symbol].append((computed_at, horizons_json, price))
        price_points[symbol].append((computed_at, price))
    prices = {sym: PriceSeries(pts) for sym, pts in price_points.items()}

    positioning: Dict[str, PriceSeries] = {}
    if _table_exists(conn, "positioning_history"):
        pos_points: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for symbol, ts, glob in conn.execute(
            "SELECT symbol, timestamp, global_account_ratio FROM positioning_history"
            " WHERE global_account_ratio IS NOT NULL ORDER BY timestamp"
        ).fetchall():
            sig = long_fraction_signal(ratio_to_long_fraction(glob))
            if sig is not None:
                pos_points[symbol].append((ts, sig))
        positioning = {sym: PriceSeries(pts) for sym, pts in pos_points.items()}

    funding: Dict[str, PriceSeries] = {}
    if _table_exists(conn, "derivatives_history"):
        from sentiment_engine.storage.derivatives_history import funding_signal as _funding_signal
        fund_points: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        for symbol, ts, fr in conn.execute(
            "SELECT symbol, timestamp, funding_rate FROM derivatives_history"
            " WHERE funding_rate IS NOT NULL ORDER BY timestamp"
        ).fetchall():
            sig = _funding_signal(fr)
            if sig is not None:
                fund_points[symbol].append((ts, sig))
        funding = {sym: PriceSeries(pts) for sym, pts in fund_points.items()}
    conn.close()
    return dict(outlooks), prices, positioning, funding


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the outlook calibration model.")
    parser.add_argument("--db", default=JOURNAL_DB_PATH, help="Journal DB (live or a snapshot).")
    parser.add_argument("--out", default=CALIBRATION_MODEL_PATH, help="Model output path.")
    parser.add_argument("--dry-run", action="store_true", help="Print the report, do not write the model.")
    args = parser.parse_args()

    outlooks, prices, positioning, funding = load_journal(args.db)
    n_outlooks = sum(len(v) for v in outlooks.values())
    n_pos = sum(len(p._t) for p in positioning.values()) if positioning else 0
    n_fund = sum(len(p._t) for p in funding.values()) if funding else 0
    print(f"loaded {n_outlooks} outlooks across {len(outlooks)} symbols from {args.db}"
          f" ({n_pos} positioning, {n_fund} funding points)")

    model = train(outlooks, prices, positioning_by_symbol=positioning or None,
                  funding_by_symbol=funding or None)

    print(f"\n=== calibration report (adopted overall: {model.adopted}) ===")
    for h, rep in model.report.get("horizons", {}).items():
        line = f"  horizon {int(h)//3600}h: n={rep['n_samples']:>5}"
        if "n_independent_oos" in rep:
            line += f" indep_oos={rep['n_independent_oos']:>4}"
        if rep.get("oos_accuracy_learned") is not None:
            line += f" learned={rep['oos_accuracy_learned']:.3f} default={rep['oos_accuracy_default']:.3f}"
        line += f"  -> {'ADOPTED' if rep['adopted'] else 'kept default'}  ({rep['reason']})"
        print(line)
    lsr = model.report.get("lsr", {})
    if lsr:
        extra = ""
        if "n_independent_oos" in lsr:
            extra = f" indep_oos={lsr['n_independent_oos']} oos_acc={lsr.get('oos_accuracy')} {lsr.get('direction','')}"
        status = f"ADOPTED coef={model.lsr_coef}" if lsr.get("adopted") else "kept default"
        print(f"  positioning (LSR):{extra}  -> {status}  ({lsr['reason']})")
    fund = model.report.get("funding", {})
    if fund:
        extra = ""
        if "n_independent_oos" in fund:
            extra = f" indep_oos={fund['n_independent_oos']} oos_acc={fund.get('oos_accuracy')} {fund.get('direction','')}"
        status = f"ADOPTED coef={model.funding_coef}" if fund.get("adopted") else "kept default"
        print(f"  funding rate:{extra}  -> {status}  ({fund['reason']})")
    logi = model.report.get("logistic", {})
    if logi:
        extra = ""
        if "n_independent_oos" in logi:
            extra = (f" indep_oos={logi['n_independent_oos']}"
                     f" logistic={logi.get('oos_accuracy_logistic')} baseline={logi.get('oos_accuracy_baseline')}")
        status = "ADOPTED (multi-feature model now drives the outlook)" if logi.get("adopted") else "kept default"
        print(f"  logistic (all features):{extra}  -> {status}  ({logi['reason']})")

    if args.dry_run:
        print("\n--dry-run: model not written")
        return
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(model.to_json())
    print(f"\nwrote {args.out}")
    if not model.adopted:
        print("No horizon cleared the gate yet -> the live scorer keeps its default "
              "coefficients. Re-run after more matured data accumulates.")


if __name__ == "__main__":
    main()

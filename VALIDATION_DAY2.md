# Sentiment Engine — Day-2 Validation

_Window: ~2.6 days (Jun 11 23:13 → Jun 14 15:37 UTC). 4,137 outlooks, 409k price points.
Source: recovered.db (salvaged after journal corruption — see note at bottom)._

## Headline: no demonstrated edge yet, and a calibration problem to fix first

The high hit-rates the raw report throws out (ADA 99–100%, BTC 87% @24h) are **artifacts**, not skill.
Two things create them: (a) outlooks are written every ~55s, so 24h windows are ~99% overlapping —
there are only ~2–3 *independent* 24h observations in 2.6 days; and (b) per-symbol scores are
almost entirely one-sided, so "hit-rate" mostly measures the market's drift, not the prediction.

## What the data actually shows

**The outlook is stuck-sign per coin — and both standing calls were wrong.**

| Symbol | Outlooks | Composite sign | Actual 2-day move | Verdict |
|---|---|---|---|---|
| BTC/USDT | 3,982 | 97% negative (bearish) | **+1.04%** | wrong direction |
| ADA/USDT | 155 | 98% positive (bullish) | **−2.11%** | wrong direction |

The engine wasn't reacting to changing conditions; it held essentially one sign per symbol for
two days. That points to a **threshold / calibration issue**, not a data-volume issue.

**Only the 1h horizon has enough independent samples to read, and it shows ~zero signal so far:**
BTC 1h information-coefficient ≈ −0.03, 6h ≈ +0.03. The 24h IC of +0.16 sits on ~2–3 independent
windows — statistically meaningless at this stage.

**ADA is under-sampled:** 155 outlooks over 43h vs BTC's 3,982 over 63h. ADA outlook generation is
sparse/intermittent — worth investigating (feed gap or too few events to trigger an outlook).

## So "what do we do to keep learning"

1. **Fix the one-sided bias first (highest value).** Diagnose why composite is pinned negative on BTC /
   positive on ADA — likely a baseline/threshold in the outlook scorer or a dominant always-on
   component (context_tilt?). A predictor that never changes sign can't be validated.
2. **Keep collecting, but judge on 1h for now.** 6h/24h need ~2–3 weeks before the numbers mean
   anything. Don't tune them on this sample.
3. **Investigate ADA sparsity** so both symbols produce comparable outlook density.
4. **Only then** consider adding velocity sources (X/cashtags). More data won't help a miscalibrated
   scorer — and there's no edge yet that extra cost would amplify.

> ⚠️ This is a measurement of a 2.6-day sample on a system that does not trade. It shows the current
> outlook has **no validated predictive value** and was directionally wrong over the window. Treat it
> as a calibration finding, not a trading signal.

## Data-integrity note
The live journal corrupted again (shared-folder torn writes). 100% of rows were salvaged via raw
B-tree page recovery into `recovered.db` (`integrity: ok`). Root-cause fix pending: move the engine's
DB off the synced folder.

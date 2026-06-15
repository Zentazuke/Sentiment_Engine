# Crypto Sentiment & Market Intelligence Engine

Standalone engine that watches live market microstructure (Binance), news,
Reddit, and global market context, then produces:

- **confirm / neutral / veto** evaluations of existing LONG/SHORT signals (scalping layer)
- **1h / 6h / 24h sentiment outlooks** per symbol (short/medium-term layer)
- **panic / euphoria shock alerts**

It **never trades and never generates trade signals.** Everything is journaled
to SQLite so its predictive value can be measured.

Symbols: `BTC/USDT`, `ADA/USDT`, `ETH/USDT`, `BNB/USDT` (configurable via `SENTIMENT_SYMBOLS`).

### Where the data lives
The live journal is stored on **local disk** (`%LOCALAPPDATA%\sentiment_engine\sentiment_journal.db`),
NOT in this synced project folder — running SQLite on a cloud-synced folder caused
torn writes and DB corruption. Every 10 min the live feeds write a consistent,
read-only **snapshot** into the project folder as `sentiment_journal_snapshot.db`
(override paths with `SENTIMENT_JOURNAL_DB` / `SENTIMENT_SNAPSHOT_DB`,
interval with `SENTIMENT_SNAPSHOT_INTERVAL`). Inspect either with
`python check_db.py` (live) or `python check_db.py sentiment_journal_snapshot.db`.

---

## Daily startup

**Easiest:** double-click `start_sentiment.bat` — it starts everything below,
opens the dashboard in your browser, and offers to run the backfill.

## Manual startup (3 terminals)

**Terminal 1 — engine API**
```powershell
cd C:\Users\Ricardo\Desktop\Work\PROJECTOS\sentiment_engine_independent_v1
python -m uvicorn sentiment_engine.api:app --host 127.0.0.1 --port 8787
```
Wait for `Application startup complete`.

**Terminal 2 — live feeds (Binance microstructure + news + context + long/short + derivatives)**
```powershell
cd C:\Users\Ricardo\Desktop\Work\PROJECTOS\sentiment_engine_independent_v1
python -m sentiment_engine.ingestion.run_live --sources binance news bluesky context lsr deriv --symbols BTC/USDT ADA/USDT ETH/USDT BNB/USDT
```
> **`bluesky`** streams Bluesky's free, keyless Jetstream firehose and keeps posts
> mentioning a coin (cashtag or full name only — no bare tickers), pushing them as
> source `bluesky`. This refills retail-sentiment volume with no signup. Nothing to
> configure.
>
> **`lsr`** pulls Binance's keyless crowd long/short ratios (global + top-trader
> account/position + taker buy/sell) per coin. The calibrator learns whether crowd
> positioning is momentum or contrarian from realized outcomes, applied once adopted.
>
> **`deriv`** adds Binance funding rate (leverage cost — persistently positive =
> crowded longs) and open interest (conviction) per coin, in the dashboard's Crowd
> Positioning & Derivatives band. Funding is wired into the calibrator as a learned
> coefficient (same adopt-gate as positioning).
>
> **CryptoPanic** is pulled via its keyless RSS feed (already in the news source
> list) — no token. Its free API is being discontinued and the crowd votes are
> now paywalled, so RSS (headline text, scored by our own model) is the path. The
> token-based `cryptopanic` connector still exists for anyone on a paid API plan.
Look for `trade feed: connected` and `book feed: connected`.

### Optional: transformer sentiment model (CryptoBERT / FinBERT)
By default text is scored by the built-in lexicon. To use a transformer instead:
```powershell
pip install transformers torch
set SENTIMENT_TRANSFORMER_MODEL=ElKulako/cryptobert   # or ProsusAI/finbert for news
```
The engine auto-falls back to the lexicon if the model can't load, so this is safe
to toggle. CryptoBERT is tuned on crypto social text; FinBERT on financial news.

**Low-resource servers — use ONNX int8 instead (no PyTorch).** The torch build needs
~1.5–2 GB RAM; an int8-quantized ONNX model needs only ~300–400 MB and `onnxruntime`
+ `tokenizers` at runtime. Convert once on a machine that has torch, then ship the
result to the server:
```powershell
pip install "optimum[onnxruntime]" transformers torch
python convert_cryptobert_onnx.py            # writes ./cryptobert_onnx/ (model + tokenizer)
```
On the server (`pip install -r requirements-onnx.txt`, no torch) set the three vars it
prints:
```
set SENTIMENT_ONNX_MODEL=.../cryptobert_onnx/model_quantized.onnx
set SENTIMENT_ONNX_TOKENIZER=.../cryptobert_onnx/tokenizer.json
set SENTIMENT_ONNX_LABELS=bearish,neutral,bullish
```
Scoring priority is ONNX → torch transformer → lexicon, each falling back automatically.

> **Reddit:** as of June 2026 Reddit's API requires approved OAuth and its
> policy restricts feeding data to ML models, so Reddit is no longer a default
> source. News is the primary social source (12 RSS feeds, incl. Cardano-
> specific). If you obtain a Reddit app, set `SENTIMENT_REDDIT_CLIENT_ID` /
> `SENTIMENT_REDDIT_CLIENT_SECRET` and add `reddit` back to `--sources`.

**Terminal 3 — dashboard**
```powershell
cd C:\Users\Ricardo\Desktop\Work\PROJECTOS\sentiment_dashboard_react\sentiment_dashboard_react
npm run dev
```
Open http://127.0.0.1:5173 → click **Check health** → enable **Auto-refresh 2s**.

### First run / after a long pause: backfill history (optional but recommended)
Populates the 6h/24h outlook windows immediately from dated news/Reddit
backlogs instead of waiting a day. Engine must be running. Safe to re-run
(duplicates are ignored).
```powershell
python -m sentiment_engine.ingestion.backfill --hours 48
```

### Learning loop (outlook calibration)
The outlook ships with hand-set coefficients. The calibrator *learns* them from
journaled outcomes — for every matured outlook it knows the realized forward
return, so it refits the sentiment->direction mapping per horizon.
```powershell
python -m sentiment_engine.signals.calibrate --dry-run    # report only
python -m sentiment_engine.signals.calibrate              # writes calibration_model.json
```
It is deliberately conservative (this feeds a trading-adjacent tool):
ridge-regularized toward the current defaults, **walk-forward** out-of-sample
evaluated, scored on **de-overlapped** (near-independent) samples, and the
learned model is **adopted only if it beats the default OOS by a margin** with
enough independent samples. Until then the live scorer keeps its defaults and
the report says why. The engine hot-reloads `calibration_model.json` when it
changes — no restart needed. Re-run as data accumulates (a few weeks gives the
6h/24h horizons enough independent windows to mean anything).

Three things are learned, each behind its own adopt-gate: the per-horizon
sentiment weights, scalar **positioning (LSR)** and **funding** coefficients, and
a multi-feature **logistic model** (`logistic_model.py`) that pools every feature
(per-horizon sentiment + trend, positioning, funding) to predict 6h direction.
When the logistic model clears its (stricter, ≥80 independent samples) gate it
*supersedes* the linear blend for the live outlook; until then everything falls
back to the defaults. All of it is pure-Python — no ML dependency.

### One-time setup (new machine)
```powershell
pip install -r requirements.txt
```

---

## Analysis tools (run anytime, engine can be off)

```powershell
# Microstructure metric distributions -> calibrate confirm/veto thresholds
python -m sentiment_engine.storage.micro_stats --symbol BTC/USDT --hours 24

# Did the outlook predict anything? Hit-rate & correlation vs realized returns
python -m sentiment_engine.storage.outlook_report --symbol BTC/USDT --days 7

# Run the full test suite
python -m pytest sentiment_engine/tests/ -q
```

---

## API endpoints (http://127.0.0.1:8787)

| Endpoint | What it does |
|---|---|
| `GET /health` | Engine status + per-symbol feed freshness |
| `GET /snapshot/{BTC-USDT}` | Short-term sentiment + microstructure block |
| `GET /outlook/{BTC-USDT}` | 1h/6h/24h outlook, composite score, context |
| `GET /alerts/{BTC-USDT}` | Panic/euphoria shock detection |
| `GET /context` | Fear & Greed, BTC dominance, market cap 24h |
| `GET /social/{BTC-USDT}/recent` | Latest news/social events with scores |
| `POST /evaluate` | confirm/neutral/veto for a STRAT_LONG/STRAT_SHORT signal |
| `POST /ingest/social` `/price` `/microstructure` `/context` | Data ingestion (used by feeds/dashboard) |

---

## X/Twitter connector (dormant until you add a key)

Built and tested, but X reads cost money (pay-per-use ~$0.005/tweet). To
activate: create a developer key at console.x.com, then:

```powershell
$env:SENTIMENT_X_BEARER_TOKEN="your-token"
python -m sentiment_engine.ingestion.run_live --sources binance news reddit context x --symbols BTC/USDT ADA/USDT
```

A hard daily budget protects your bill: `SENTIMENT_X_DAILY_READS` (default
400 reads/day ≈ $2/day). Tweets arrive as source `x:search` and flow through
the same scoring, outlook, and shock pipeline.

## Trading bot shadow mode

The bot (separate project) fires `evaluate_async` on every signal via
`sentiment_shadow_client.py` — fire-and-forget, never read, never blocking.
Disable with env `SENTIMENT_SHADOW=off` on the bot side. Engine answers are
journaled with price outcomes 30s/2m/5m later; compare before ever enabling
veto mode.

## Safety rules (non-negotiable)

1. Never places trades, never generates buy/sell signals.
2. Bad/missing/stale data → neutral, with reduced confidence.
3. Sentiment can nudge confidence but never flips a microstructure decision.
4. Every evaluation and outlook is journaled.

## Troubleshooting

- **Dashboard alert "HTTP 500"** → the engine isn't running (the Vite proxy
  reports 500 when it can't reach port 8787). Start terminal 1.
- **"No feed" on symbols** → terminal 2 isn't running or can't reach Binance.
  Geo-blocked? Set `SENTIMENT_BINANCE_WS_URL=wss://stream.binance.us:9443/stream`.
- **Rel. Volume shows "—"** → normal for the first 5 minutes (baseline warmup).
- **Outlook says "insufficient data"** → run the backfill, keep feeds running.
- **`ModuleNotFoundError`** → `pip install -r requirements.txt`.
- **Engine commands fail** → you're probably in the dashboard folder; engine
  commands run from `sentiment_engine_independent_v1`.

## Key environment variables (all optional)

`SENTIMENT_SYMBOLS`, `SENTIMENT_PORT`, `SENTIMENT_JOURNAL_DB`,
`SENTIMENT_BINANCE_WS_URL`, `SENTIMENT_NEWS_RSS_URLS`,
`SENTIMENT_REDDIT_SUBREDDITS`, `SENTIMENT_MICRO_*` (decision thresholds —
calibrate with micro_stats before changing), `SENTIMENT_OUTLOOK_*`,
`SENTIMENT_SHOCK_*`, `SENTIMENT_CONTEXT_*`. Defaults are in
`sentiment_engine/config.py`, every value is env-overridable.

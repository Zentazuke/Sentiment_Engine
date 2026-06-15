# Sentiment Engine — Improvement Research

_Prioritized roadmap for more information (data sources) and better models, tailored to the
current engine. Researched June 2026._

## TL;DR — highest return, in order

1. **Binance funding rate + open interest** — free, keyless, same `fapi` host the long/short feed
   already uses, and it works from your machine (the LSR feed proves it). Drop-in like `lsr`. Funding
   = how expensive it is to be long (leverage sentiment); OI = conviction behind a move. This is the
   single best effort-to-value addition.
2. **CryptoPanic API** — free tier, gives a crowd **bullish/bearish vote** and "Panic Score" per news
   item across a huge source list. It's a sentiment *label*, not just text, and a clean Reddit-free
   way to widen news coverage.
3. **CryptoBERT model** — swap/augment the keyword scorer with a transformer fine-tuned on 2M crypto
   social posts. This is the real "better model" step; the lexicon can't read context, sarcasm, or
   novel phrasing.
4. **A free social stream to replace Reddit** — StockTwits (explicit bullish/bearish tags) or Bluesky
   (open firehose, no auth, no ToS gray area). This refills the retail-sentiment volume the learner is
   currently starved of.

The binding constraint is still **real sentiment volume**, not model cleverness — so 1, 2, 4 (more
signal) matter at least as much as 3 (better scoring). Garbage-in beats a fancy model on thin data.

---

## Current state (what we're improving from)

- **Sources:** Binance microstructure (trades + book), news RSS (≈11 feeds incl. coin-targeted), market
  context (Fear & Greed, CoinGecko global), Binance long/short ratio. Reddit blocked by policy; X paid
  and dormant.
- **Scorer:** `sentiment_fast.py` — a good rule-based lexicon (negation, intensifiers, hedging, emoji),
  but no contextual/ML understanding.
- **Learning:** walk-forward ridge calibration per horizon + a learned positioning coefficient, with an
  adopt-only-if-it-beats-default gate.
- **Coins:** BTC, ETH, ADA, BNB.

---

## Part A — More information (new data sources)

| Source | What it adds | Cost / access | Effort | Fit |
|---|---|---|---|---|
| Binance funding rate | Leverage sentiment (longs paying = crowded long) | Free, keyless (`/fapi/v1/fundingRate`) | Low | Drop-in feed |
| Binance open interest | Conviction / position build vs unwind (`/futures/data/openInterestHist`) | Free, keyless | Low | Drop-in feed |
| CryptoPanic | Crowd bullish/bearish vote + Panic Score across many sources | Free tier (token) | Low | New social/news feed |
| StockTwits | Retail messages with explicit bullish/bearish tags (crypto cashtags) | Free basic, **verify ToS** | Medium | Reddit replacement |
| Bluesky firehose | Open, free, no-auth stream of all posts — filter crypto cashtags | Free, no key, ToS-clean | Medium-High | Reddit/X replacement |
| Google Trends | Search-interest attention proxy | Free (`pytrends`) | Low | Nice-to-have |
| On-chain flows | Exchange net-flows, stablecoin flows | Mostly paid (CryptoQuant/Glassnode) | High | Defer |
| LunarCrush Galaxy Score | Best single aggregate social score | API now **paid** (free tier dropped social/API) | Low if paid | Optional |

### Recommended now

**1. Funding rate + open interest (do first).** These are the natural siblings of the long/short ratio
you just added and use the same keyless `fapi.binance.com` host that already works for you. Funding
rate tells you *how crowded and expensive* one side is (persistently positive funding = over-leveraged
longs, a classic contrarian setup); open-interest change tells you whether a price move is backed by
*new* positioning or just short-covering. Build it exactly like `binance_lsr_feed.py` → journal to a
`derivatives_history` table → feed the calibrator as two more candidate signals (the learner decides
sign/weight). Because the plumbing mirrors what exists, this is a half-day job.

**2. CryptoPanic.** A free API token gives you a normalized news stream with a crowd bullish/bearish
vote attached to each item — so it's both *more news coverage* and *a second opinion on sentiment* that
doesn't depend on our scorer. Add it as a `news`-class source; map its vote to a sentiment value and
tag `source="cryptopanic"`.

**3. A free retail social stream (refill what Reddit gave us).**
- **StockTwits** is the most direct: users self-tag posts bullish/bearish on crypto cashtags, so you
  get labeled sentiment for free. Caveat: treat API access/ToS like we did Reddit — verify current
  terms before relying on it (their access has tightened before). Bonus: CryptoBERT was trained on
  StockTwits, so the two pair naturally.
- **Bluesky** is the cleanest long-term bet: a genuinely open, free, no-auth firehose (use the
  `Jetstream` websocket) with no approval gate and no ToS gray area. More build effort (stream +
  keyword/cashtag filter + rate handling), but it's the most future-proof social source and would give
  the learner real volume.

### Defer / optional
- **On-chain** (exchange netflow, stablecoin flows) is genuinely predictive but the good data is paid
  (CryptoQuant, Glassnode); CoinGlass has a limited free slice. Revisit once the free signals prove out.
- **LunarCrush** Galaxy Score is excellent but its free tier no longer includes API/social data — only
  worth it if you'll pay, though they do offer an MCP connector.
- **Google Trends** is free and trivial via `pytrends` — a noisy but real attention proxy; add it if you
  want breadth, low priority.

---

## Part B — Better sentiment models

The lexicon is fast and transparent but blind to context ("this is *not* the dump everyone feared" reads
bearish to it). Upgrade path:

**1. CryptoBERT (`ElKulako/cryptobert`) — the main upgrade.** A transformer fine-tuned on ~2M labeled
crypto StockTwits posts, outputting bullish/neutral/bearish. ~58% three-class accuracy on crypto social
(genuinely strong — crypto social is noisy and sarcastic). Free on Hugging Face, runs locally on CPU at
your volume. Use it as the scorer for **social** text.

**2. FinBERT (`ProsusAI/finbert`) — for news.** Tuned for formal financial language, so route **news
headlines → FinBERT** and **social → CryptoBERT**; they're strong on different registers.

**3. Keep the lexicon as a fast fallback + ensemble.** Don't rip it out: it's instant, dependency-free,
and gives a useful confidence signal. Best design is an **ensemble** — lexicon for the realtime path and
a cheap confidence prior, transformer run in **batch** over the (low-volume) news/social queue, blended
into the stored sentiment. This keeps the "cheap by default, never blocks a decision" philosophy.

**Implementation notes / cautions:**
- Adds `transformers` + a runtime (`torch`/`onnxruntime`) — heavier deps. Mitigate: env-gated optional
  scorer, batch-process (never on the 1-second microstructure path), and consider a distilled/ONNX build
  for speed. Default stays lexicon; transformer is an opt-in that overwrites the stored sentiment with a
  better score.
- An **LLM-based scorer** (call a small model for ambiguous items) is the most accurate option but adds
  cost/latency — reserve it for low-volume, high-value items if at all.

---

## Part C — Better learning (the meta-model)

Once the inputs above are flowing and a few weeks of data exist, upgrade the calibrator itself:

1. **Multi-feature model.** Move from per-horizon ridge to a regularized **logistic / gradient-boosted**
   model over the full feature set: sentiment (news vs social split), positioning (LSR), funding, OI
   change, attention/velocity, divergence, context. Literature consistently finds XGBoost/GBM over
   sentiment + technicals beats sentiment alone — but **only with enough data**; GBMs overfit hard on
   thin samples, so keep the regularized-linear model until samples are deep, and keep the walk-forward
   adopt-gate either way.
2. **Richer features:** sentiment velocity/acceleration, news-vs-retail divergence, source disagreement,
   and z-scores vs a rolling baseline (de-bias the stuck-sign problem we saw early on).
3. **Regime awareness:** condition behavior on a volatility regime (trending vs choppy). Both the
   research and the Kelly/positioning literature stress that the right signal weighting changes by
   regime.
4. **Confidence calibration:** map the outlook score to a calibrated probability (isotonic/Platt on
   realized outcomes) so "confidence 0.7" actually means ~70%.
5. **Pooled vs per-coin:** pool low-data coins (ADA, BNB) and specialize BTC as its history deepens.
6. **Evaluation discipline (mostly built):** keep tracking information-coefficient over time, hit-rate by
   regime, and lead-lag (does sentiment lead price at each horizon) on de-overlapped samples.

---

## Suggested sequencing

- **Week 1:** funding rate + OI feed (mirror the LSR feed) → wire into the calibrator. CryptoPanic feed.
- **Week 2:** pick one free social stream (StockTwits fastest, Bluesky cleanest) and start collecting.
- **Week 3:** add CryptoBERT/FinBERT as an optional batch scorer; A/B its sentiment vs the lexicon on the
  journaled outcomes.
- **Ongoing:** let the nightly calibrator decide what earns adoption; revisit the multi-feature/GBM model
  only once independent sample counts are deep enough to support it.

---

## Sources
- CryptoPanic API & community sentiment voting — https://cryptopanic.com/developers/ ; https://blockspot.io/company/cryptopanic/
- CryptoBERT (2M StockTwits, fine-tuned on FinBERT) — https://huggingface.co/ElKulako/cryptobert ; FinBERT — https://huggingface.co/ProsusAI/finbert
- Comparative crypto sentiment model study — https://www.mdpi.com/2504-2289/8/6/63
- StockTwits free/no-auth sentiment stream — https://articles.dailytickers.com/series/finance-apis/part3-sentiment/ ; https://docs.stockapis.com/parsers/stocktwits/intro
- Binance funding rate / open interest endpoints — https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History ; https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics
- Bluesky open firehose / Jetstream — https://docs.bsky.app/docs/advanced-guides/firehose
- LunarCrush Galaxy Score & API tiers — https://lunarcrush.com/ ; https://medium.com/lunarcrush/lunarcrush-api-v3-is-now-available-426148edb826
- On-chain exchange/stablecoin netflow metrics — https://userguide.cryptoquant.com/cryptoquant-metrics/exchange/exchange-in-outflow-and-netflow ; https://www.coinglass.com/CryptoApi
- Sentiment + gradient boosting for crypto price direction — https://arxiv.org/pdf/1805.00558 ; sentiment + technicals — https://arxiv.org/pdf/2410.14532 ; ML model comparison — https://pmc.ncbi.nlm.nih.gov/articles/PMC12571449/

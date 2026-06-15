# Sentiment Engine — Standalone MVP

This package is a standalone sentiment confirmation layer for the crypto scalping bot.

It **does not trade**. It only evaluates an existing bot signal and returns:

- `confirm`
- `neutral`
- `veto`

The first version uses manual/mock data and rule-based sentiment only. No X/Twitter, Reddit, Redis, LLMs, or execution changes are included yet.

## Install minimal dependencies

From the project root:

```bash
pip install fastapi uvicorn pydantic
```

## Run the local API

```bash
uvicorn sentiment_engine.api:app --host 127.0.0.1 --port 8787
```

## Check health

```bash
python -m sentiment_engine.cli health
```

## Inject a bullish scenario

```bash
python -m sentiment_engine.cli simulate --symbol BTC/USDT --scenario bullish_breakout --count 20 --baseline --price 104000 --price-change-pct 0.3
```

## Evaluate a LONG

```bash
python -m sentiment_engine.cli evaluate --symbol BTC/USDT --direction STRAT_LONG --bot-confidence 0.70 --trigger-price 104250
```

## Evaluate a SHORT

```bash
python -m sentiment_engine.cli evaluate --symbol BTC/USDT --direction STRAT_SHORT --bot-confidence 0.70 --trigger-price 104250
```

## Endpoints

- `GET /health`
- `POST /ingest/social`
- `POST /ingest/price`
- `GET /snapshot/{symbol_key}` such as `/snapshot/BTC-USDT`
- `POST /evaluate`

## Safety rules

- Low data quality returns `neutral`.
- Errors return `neutral`.
- This service never creates trades.
- This service never executes orders.
- Phase 1 should be tested independently with mock/manual data only.

## Journal

Every `/evaluate` call is logged into `sentiment_journal.db` in the current working directory.

This is what lets you compare later:

- trades sentiment would confirm
- trades sentiment would veto
- actual outcome of the bot trade

## One-command independent demo

```bash
python -m sentiment_engine.demo
```

See `INDEPENDENT_TEST_GUIDE.md` for the recommended test flow.

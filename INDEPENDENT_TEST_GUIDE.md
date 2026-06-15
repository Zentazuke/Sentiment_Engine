# Independent Test Guide — Sentiment Engine v1

This package is designed to work by itself before being connected to the trading bot.

It does **not** read the bot, modify the bot, place orders, or access real exchange accounts.
It only ingests mock/manual social and price events, builds short rolling windows, and evaluates fake LONG/SHORT signals.

## 1. Install dependencies

From the folder that contains `sentiment_engine/`:

```bash
pip install -r requirements.txt
```

Or minimally:

```bash
pip install fastapi uvicorn pydantic pytest
```

## 2. Run tests

```bash
python -m pytest sentiment_engine/tests -q
```

Expected:

```text
4 passed
```

## 3. Run the one-command demo, no API server required

```bash
python -m sentiment_engine.demo
```

Expected behavior:

- The snapshot should show BTC/USDT with recent messages.
- `STRAT_LONG` should usually return `confirm`.
- `STRAT_SHORT` should usually return `veto` or `neutral`, depending on the random mock messages.

## 4. Run as a local standalone API

Terminal 1:

```bash
uvicorn sentiment_engine.api:app --host 127.0.0.1 --port 8787
```

Terminal 2:

```bash
python -m sentiment_engine.cli health
```

## 5. Simulate a bullish BTC scenario

```bash
python -m sentiment_engine.cli simulate --symbol BTC/USDT --scenario bullish_breakout --count 25 --baseline --price 104000 --price-change-pct 0.3
```

Then evaluate a fake LONG:

```bash
python -m sentiment_engine.cli evaluate --symbol BTC/USDT --direction STRAT_LONG --bot-confidence 0.70 --trigger-price 104250
```

Then evaluate a fake SHORT:

```bash
python -m sentiment_engine.cli evaluate --symbol BTC/USDT --direction STRAT_SHORT --bot-confidence 0.70 --trigger-price 104250
```

## 6. Simulate a bearish BTC scenario

```bash
python -m sentiment_engine.cli simulate --symbol BTC/USDT --scenario panic_dump --count 25 --baseline --price 104000 --price-change-pct -0.3
```

Evaluate a fake SHORT:

```bash
python -m sentiment_engine.cli evaluate --symbol BTC/USDT --direction STRAT_SHORT --bot-confidence 0.70 --trigger-price 103700
```

## 7. What counts as success at this stage?

The independent engine is ready for the next stage when:

1. It starts reliably.
2. It accepts manual/mock social events.
3. It accepts manual/mock price events.
4. It returns `neutral` when data quality is low.
5. It confirms LONG when sentiment velocity is strongly bullish.
6. It confirms SHORT when sentiment velocity is strongly bearish.
7. It journals every `/evaluate` call into `sentiment_journal.db`.

## 8. What this version intentionally does not include

- No trading bot integration.
- No bot log watcher.
- No X/Twitter API.
- No Reddit API.
- No LLM calls.
- No Redis.
- No live order execution.
- No position sizing changes.

This is intentional. First we prove the sentiment logic works alone.

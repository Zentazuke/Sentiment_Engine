"""One-command local demo for the independent sentiment engine.

Run from the project root:
    python -m sentiment_engine.demo

This does not start a web server and does not interact with any trading bot.
It injects mock social/price events directly into the in-memory engine, then
prints the current snapshot and LONG/SHORT evaluations.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from typing import Any

from sentiment_engine.aggregation.state_store import STATE
from sentiment_engine.ingestion.mock_social import generate_messages
from sentiment_engine.processing.sentiment_fast import score_text
from sentiment_engine.signals.decision import evaluate_trade_signal
from sentiment_engine.types import PriceEvent, SocialEvent, TradeDirection


def _floatify(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if is_dataclass(value):
        return _floatify(asdict(value))
    if isinstance(value, dict):
        return {k: _floatify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_floatify(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _add_mock_social(symbol: str, scenario: str, count: int, seconds: int) -> None:
    for msg in generate_messages(scenario=scenario, symbol=symbol, count=count, seconds=seconds):
        sentiment, confidence = score_text(msg.text)
        STATE.add_social(
            SocialEvent(
                symbol=msg.symbol,
                source=msg.source,
                text=msg.text,
                timestamp=msg.timestamp,
                sentiment=sentiment,
                confidence=confidence,
            )
        )


def _add_price_path(symbol: str, start: Decimal, end: Decimal, steps: int = 8) -> None:
    now = time.time()
    for idx in range(steps):
        pct = Decimal(idx) / Decimal(max(1, steps - 1))
        price = start + ((end - start) * pct)
        STATE.add_price(PriceEvent(symbol=symbol, price=price, timestamp=now - (steps - idx)))


def main() -> int:
    symbol = "BTC/USDT"

    # Create a calmer previous window, then a stronger recent bullish window.
    _add_mock_social(symbol, "mixed_chop", count=12, seconds=110)
    _add_mock_social(symbol, "bullish_breakout", count=30, seconds=25)
    _add_price_path(symbol, Decimal("104000"), Decimal("104350"))

    snapshot = STATE.snapshot(symbol)
    long_eval = evaluate_trade_signal(
        snapshot=snapshot,
        direction=TradeDirection.LONG,
        bot_confidence=Decimal("0.70"),
    )
    short_eval = evaluate_trade_signal(
        snapshot=snapshot,
        direction=TradeDirection.SHORT,
        bot_confidence=Decimal("0.70"),
    )

    print("=== Snapshot ===")
    print(json.dumps(_floatify(snapshot), indent=2, sort_keys=True))
    print("\n=== Evaluate STRAT_LONG ===")
    print(json.dumps(_floatify(long_eval), indent=2, sort_keys=True))
    print("\n=== Evaluate STRAT_SHORT ===")
    print(json.dumps(_floatify(short_eval), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

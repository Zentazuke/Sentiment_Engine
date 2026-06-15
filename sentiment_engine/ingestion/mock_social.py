"""Mock social scenarios for standalone testing."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True, slots=True)
class MockMessage:
    symbol: str
    source: str
    text: str
    timestamp: float


SCENARIOS = {
    "bullish_breakout": [
        "BTC breakout looks strong, bulls taking control",
        "BTC pumping hard, support held perfectly",
        "Massive squeeze incoming, buyers stepping in",
        "Clean reclaim, momentum looks bullish",
    ],
    "panic_dump": [
        "BTC dumping hard, support lost",
        "Panic selloff, longs getting rekt",
        "Bearish rejection, buyers disappeared",
        "Crash vibes, this looks ugly",
    ],
    "mixed_chop": [
        "BTC might break out but not sure",
        "Choppy price action, no clean direction",
        "Some buyers but rejection risk is high",
        "Market feels undecided right now",
    ],
    "euphoria_top": [
        "BTC moon mission, parabolic pump",
        "Everyone is bullish, insane green candles",
        "This is going vertical, no bears left",
        "Massive pump, send it higher",
    ],
    "bearish_continuation": [
        "Another rejection, bearish continuation likely",
        "Support lost, short setup looks clean",
        "Sellers still in control, red candles stacking",
        "Weak bounce, likely more downside",
    ],
}


def generate_messages(
    *,
    scenario: str,
    symbol: str,
    count: int,
    seconds: int = 30,
) -> Iterable[MockMessage]:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}. Options: {sorted(SCENARIOS)}")
    now = time.time()
    phrases: List[str] = SCENARIOS[scenario]
    for index in range(count):
        # Spread events over the requested period but keep them recent.
        ts = now - max(0, seconds - index * (seconds / max(count, 1)))
        yield MockMessage(
            symbol=symbol,
            source="mock",
            text=random.choice(phrases),
            timestamp=ts,
        )

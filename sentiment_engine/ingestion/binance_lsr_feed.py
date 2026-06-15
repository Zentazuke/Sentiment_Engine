"""Binance long/short ratio feed (keyless futures data, read-only).

Polls four public Binance USD-M futures "long/short ratio" endpoints per symbol
and pushes the latest values through the engine's /ingest/positioning endpoint.
This is crowd positioning (how accounts/positions are leaning long vs short),
not text sentiment -- the calibration learner decides whether it is momentum or
contrarian.

Endpoints (no API key required, last 30 days, ~5 min granularity):
  /futures/data/globalLongShortAccountRatio   (all accounts)
  /futures/data/topLongShortAccountRatio       (top traders, by account)
  /futures/data/topLongShortPositionRatio      (top traders, by position)
  /futures/data/takerlongshortRatio            (taker buy/sell volume)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, List, Optional, Sequence

from sentiment_engine.config import LSR_BASE_URL, LSR_PERIOD, LSR_POLL_SECONDS
from sentiment_engine.processing.coin_mapper import normalize_symbol

logger = logging.getLogger(__name__)

_RATIO_PATHS = {
    "global_account_ratio": "/futures/data/globalLongShortAccountRatio",
    "top_account_ratio": "/futures/data/topLongShortAccountRatio",
    "top_position_ratio": "/futures/data/topLongShortPositionRatio",
}
_TAKER_PATH = "/futures/data/takerlongshortRatio"


def futures_symbol(symbol: str) -> str:
    """'BTC/USDT' -> 'BTCUSDT'."""
    return normalize_symbol(symbol).replace("/", "")


def parse_latest(payload: Any, field: str) -> Optional[float]:
    """Last element's `field` parsed as float, or None if unusable."""
    try:
        if not isinstance(payload, list) or not payload:
            return None
        value = payload[-1].get(field)
        return float(value) if value is not None else None
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


class BinanceLsrFeed:
    def __init__(
        self,
        symbols: Sequence[str],
        engine_url: str,
        base_url: str = LSR_BASE_URL,
        period: str = LSR_PERIOD,
        poll_seconds: float = LSR_POLL_SECONDS,
    ) -> None:
        self.symbols = [normalize_symbol(s) for s in symbols]
        self.engine_url = engine_url.rstrip("/")
        self.base_url = base_url.rstrip("/")
        self.period = period
        self.poll_seconds = poll_seconds
        self.pushed_count = 0

    async def _fetch(self, client, path: str, fsym: str, field: str) -> Optional[float]:
        url = f"{self.base_url}{path}?symbol={fsym}&period={self.period}&limit=1"
        try:
            resp = await client.get(url, headers={"User-Agent": "sentiment-engine/0.1"})
            resp.raise_for_status()
            return parse_latest(resp.json(), field)
        except Exception as exc:  # noqa: BLE001 - best effort per endpoint
            status = getattr(getattr(exc, "response", None), "status_code", None)
            logger.warning("lsr feed: %s %s failed (%s status=%s)", fsym, path, type(exc).__name__, status)
            return None

    async def run(self) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            logger.info("lsr feed: polling long/short ratios for %s every %.0fs (period %s)",
                        ", ".join(self.symbols), self.poll_seconds, self.period)
            while True:
                for symbol in self.symbols:
                    fsym = futures_symbol(symbol)
                    ratios = {key: await self._fetch(client, path, fsym, "longShortRatio")
                              for key, path in _RATIO_PATHS.items()}
                    taker = await self._fetch(client, _TAKER_PATH, fsym, "buySellRatio")
                    if all(v is None for v in ratios.values()) and taker is None:
                        continue
                    try:
                        await client.post(
                            f"{self.engine_url}/ingest/positioning",
                            json={
                                "symbol": symbol,
                                "global_account_ratio": ratios["global_account_ratio"],
                                "top_account_ratio": ratios["top_account_ratio"],
                                "top_position_ratio": ratios["top_position_ratio"],
                                "taker_ratio": taker,
                                "timestamp": time.time(),
                            },
                        )
                        self.pushed_count += 1
                        logger.info("lsr feed: %s global=%s top_pos=%s taker=%s",
                                    symbol, ratios["global_account_ratio"],
                                    ratios["top_position_ratio"], taker)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("lsr feed: push failed (%s)", type(exc).__name__)
                await asyncio.sleep(self.poll_seconds)

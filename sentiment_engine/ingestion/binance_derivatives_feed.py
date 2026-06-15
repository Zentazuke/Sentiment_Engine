"""Binance derivatives feed (keyless futures data, read-only).

Polls two public Binance USD-M futures endpoints per symbol and pushes the
latest values through /ingest/derivatives:

  /fapi/v1/fundingRate            -> latest settled funding rate (signed)
  /futures/data/openInterestHist  -> latest open interest (USD notional)

Same keyless host as the long/short feed. Crowd/leverage signal, not text
sentiment; the calibrator will decide its weight from realized outcomes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Sequence

from sentiment_engine.config import LSR_BASE_URL, LSR_PERIOD, DERIV_POLL_SECONDS
from sentiment_engine.ingestion.binance_lsr_feed import futures_symbol, parse_latest
from sentiment_engine.processing.coin_mapper import normalize_symbol

logger = logging.getLogger(__name__)

_FUNDING_PATH = "/fapi/v1/fundingRate"
_OI_PATH = "/futures/data/openInterestHist"


class BinanceDerivativesFeed:
    def __init__(
        self,
        symbols: Sequence[str],
        engine_url: str,
        base_url: str = LSR_BASE_URL,
        period: str = LSR_PERIOD,
        poll_seconds: float = DERIV_POLL_SECONDS,
    ) -> None:
        self.symbols = [normalize_symbol(s) for s in symbols]
        self.engine_url = engine_url.rstrip("/")
        self.base_url = base_url.rstrip("/")
        self.period = period
        self.poll_seconds = poll_seconds
        self.pushed_count = 0

    async def _get(self, client, url: str, field: str) -> Optional[float]:
        try:
            resp = await client.get(url, headers={"User-Agent": "sentiment-engine/0.1"})
            resp.raise_for_status()
            return parse_latest(resp.json(), field)
        except Exception as exc:  # noqa: BLE001 - best effort per endpoint
            status = getattr(getattr(exc, "response", None), "status_code", None)
            logger.warning("derivatives feed: %s failed (%s status=%s)", url, type(exc).__name__, status)
            return None

    async def run(self) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            logger.info("derivatives feed: polling funding + open interest for %s every %.0fs",
                        ", ".join(self.symbols), self.poll_seconds)
            while True:
                for symbol in self.symbols:
                    fsym = futures_symbol(symbol)
                    funding = await self._get(
                        client, f"{self.base_url}{_FUNDING_PATH}?symbol={fsym}&limit=1", "fundingRate")
                    oi = await self._get(
                        client, f"{self.base_url}{_OI_PATH}?symbol={fsym}&period={self.period}&limit=1",
                        "sumOpenInterestValue")
                    if funding is None and oi is None:
                        continue
                    try:
                        await client.post(
                            f"{self.engine_url}/ingest/derivatives",
                            json={
                                "symbol": symbol,
                                "funding_rate": funding,
                                "open_interest_usd": oi,
                                "timestamp": time.time(),
                            },
                        )
                        self.pushed_count += 1
                        logger.info("derivatives feed: %s funding=%s oi_usd=%s", symbol, funding, oi)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("derivatives feed: push failed (%s)", type(exc).__name__)
                await asyncio.sleep(self.poll_seconds)

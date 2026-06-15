"""Market context connector (keyless, read-only).

Polls alternative.me Fear & Greed and CoinGecko global metrics every ~15 min
and pushes a ContextSnapshot to the engine. Both are free, keyless public
APIs. Failures are non-fatal: missing context simply means zero tilt.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from sentiment_engine.config import COINGECKO_GLOBAL_URL, CONTEXT_POLL_SECONDS, FNG_URL

logger = logging.getLogger(__name__)


def parse_fng(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse alternative.me /fng/ payload -> partial context fields."""
    try:
        entry = data["data"][0]
        return {
            "fear_greed_value": int(entry["value"]),
            "fear_greed_label": str(entry.get("value_classification") or ""),
        }
    except (KeyError, IndexError, TypeError, ValueError):
        return {}


def parse_coingecko_global(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse CoinGecko /global payload -> partial context fields."""
    try:
        payload = data["data"]
        out: Dict[str, Any] = {}
        dominance = payload.get("market_cap_percentage", {}).get("btc")
        if dominance is not None:
            out["btc_dominance_pct"] = float(dominance)
        mcap = payload.get("total_market_cap", {}).get("usd")
        if mcap is not None:
            out["total_market_cap_usd"] = float(mcap)
        change = payload.get("market_cap_change_percentage_24h_usd")
        if change is not None:
            out["market_cap_change_24h_pct"] = float(change)
        return out
    except (KeyError, TypeError, ValueError):
        return {}


class ContextFeed:
    """Polls keyless context sources and pushes merged snapshots to the engine."""

    def __init__(self, engine_url: str, poll_seconds: float = CONTEXT_POLL_SECONDS) -> None:
        self.engine_url = engine_url.rstrip("/")
        self.poll_seconds = poll_seconds
        self.last_push: Optional[Dict[str, Any]] = None

    async def fetch_once(self, client) -> Dict[str, Any]:
        """One merged fetch; partial results are fine."""
        merged: Dict[str, Any] = {}
        for url, parser, name in (
            (FNG_URL, parse_fng, "fear&greed"),
            (COINGECKO_GLOBAL_URL, parse_coingecko_global, "coingecko"),
        ):
            try:
                response = await client.get(url, headers={"User-Agent": "sentiment-engine/0.1"})
                response.raise_for_status()
                merged.update(parser(response.json()))
            except Exception as exc:  # noqa: BLE001 - context is best-effort
                logger.warning("context feed: %s fetch failed (%s)", name, type(exc).__name__)
        return merged

    async def run(self) -> None:
        import httpx  # local import keeps unit tests dependency-free

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            logger.info("context feed: polling every %.0fs", self.poll_seconds)
            while True:
                fields = await self.fetch_once(client)
                if fields:
                    try:
                        await client.post(f"{self.engine_url}/ingest/context", json=fields)
                        self.last_push = fields
                        logger.info("context feed: pushed %s", fields)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("context feed: push failed (%s)", type(exc).__name__)
                await asyncio.sleep(self.poll_seconds)

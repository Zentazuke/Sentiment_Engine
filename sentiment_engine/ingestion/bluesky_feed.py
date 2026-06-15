"""Bluesky social feed (keyless Jetstream firehose, read-only).

Bluesky's firehose is open and free with no auth: we connect to a public
Jetstream relay, stream every new post, keep only those mentioning one of our
coins, and push them through /ingest/social (source "bluesky"). This refills the
retail-sentiment volume lost when Reddit's API closed.

Matching is deliberately STRICT (cashtag or full coin name, never a bare ticker
like "ada"/"bnb") because the firehose is general-purpose, so bare tickers would
pull in huge amounts of unrelated chatter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sentiment_engine.config import BLUESKY_JETSTREAM_URL
from sentiment_engine.processing.coin_mapper import normalize_symbol

logger = logging.getLogger(__name__)

_MAX_TEXT = 500

# Cashtag OR full coin name only — no bare tickers, to avoid firehose noise.
_PATTERNS = {
    "BTC/USDT": re.compile(r"\$btc\b|\bbitcoin\b", re.IGNORECASE),
    "ETH/USDT": re.compile(r"\$eth\b|\bethereum\b", re.IGNORECASE),
    "ADA/USDT": re.compile(r"\$ada\b|\bcardano\b", re.IGNORECASE),
    "BNB/USDT": re.compile(r"\$bnb\b|\bbinance coin\b", re.IGNORECASE),
    "SOL/USDT": re.compile(r"\$sol\b|\bsolana\b", re.IGNORECASE),
    "XRP/USDT": re.compile(r"\$xrp\b|\bxrp\b|\bripple\b", re.IGNORECASE),
    "DOGE/USDT": re.compile(r"\$doge\b|\bdogecoin\b", re.IGNORECASE),
    "LINK/USDT": re.compile(r"\$link\b|\bchainlink\b", re.IGNORECASE),
}


def extract_post_text(event: Dict[str, Any]) -> Optional[str]:
    """Pull the text of a newly-created Bluesky post from a Jetstream event."""
    try:
        if event.get("kind") != "commit":
            return None
        commit = event["commit"]
        if commit.get("operation") != "create" or commit.get("collection") != "app.bsky.feed.post":
            return None
        text = commit["record"].get("text")
        return text if isinstance(text, str) and text.strip() else None
    except (KeyError, TypeError):
        return None


class BlueskyFeed:
    def __init__(
        self,
        symbols: Sequence[str],
        engine_url: str,
        jetstream_url: str = BLUESKY_JETSTREAM_URL,
    ) -> None:
        self.symbols = [normalize_symbol(s) for s in symbols]
        self.engine_url = engine_url.rstrip("/")
        self.jetstream_url = jetstream_url
        self._patterns = [(sym, _PATTERNS[sym]) for sym in self.symbols if sym in _PATTERNS]
        self.pushed_count = 0

    def match_symbols(self, text: str) -> List[str]:
        return [symbol for symbol, pattern in self._patterns if pattern.search(text)]

    def process_event(self, event: Dict[str, Any]) -> List[Tuple[str, str]]:
        """(symbol, text) pairs to push for one firehose event. Pure; no network."""
        text = extract_post_text(event)
        if text is None:
            return []
        matched = self.match_symbols(text)
        return [(symbol, text[:_MAX_TEXT]) for symbol in matched]

    async def run(self) -> None:
        import httpx
        import websockets  # local import keeps unit tests dependency-free

        backoff = 1.0
        async with httpx.AsyncClient(timeout=10.0) as client:
            while True:
                try:
                    logger.info("bluesky feed: connecting jetstream (%s)", ", ".join(self.symbols))
                    async with websockets.connect(self.jetstream_url, ping_interval=20, ping_timeout=20, max_size=None) as ws:
                        logger.info("bluesky feed: connected")
                        backoff = 1.0
                        async for raw in ws:
                            try:
                                event = json.loads(raw)
                            except (ValueError, TypeError):
                                continue
                            for symbol, text in self.process_event(event):
                                try:
                                    await client.post(
                                        f"{self.engine_url}/ingest/social",
                                        json={
                                            "symbol": symbol,
                                            "source": "bluesky",
                                            "text": text,
                                            "timestamp": time.time(),
                                        },
                                    )
                                    self.pushed_count += 1
                                    logger.info("bluesky feed: %s <- %r", symbol, text[:70])
                                except Exception as exc:  # noqa: BLE001
                                    logger.warning("bluesky feed: push failed (%s)", type(exc).__name__)
                except Exception as exc:  # noqa: BLE001 - reconnect boundary
                    backoff = min(backoff * 2, 60)
                    logger.warning("bluesky feed: stream dropped (%s); reconnecting in %.0fs", type(exc).__name__, backoff)
                    await asyncio.sleep(backoff)

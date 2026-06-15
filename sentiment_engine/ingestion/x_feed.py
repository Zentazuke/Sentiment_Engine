"""X/Twitter cashtag connector (key-gated, read-only, budget-capped).

Polls the X v2 recent-search API for $BTC / $ADA cashtag posts and pushes
them through /ingest/social (source "x:search").

Cost control - this API is pay-per-use (~$0.005 per tweet read):
- hard daily read budget (SENTIMENT_X_DAILY_READS, default 400 ≈ $2/day)
- since_id tracking so the same tweet is never paid for twice
- small max_results per poll, long poll interval

Without SENTIMENT_X_BEARER_TOKEN the feed refuses to start (and run_live
skips it with a clear message). Like every connector: read-only, fail-safe,
never trades.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sentiment_engine.config import (
    X_BEARER_TOKEN,
    X_DAILY_READ_BUDGET,
    X_MAX_RESULTS,
    X_POLL_SECONDS,
    X_SEARCH_URL,
)
from sentiment_engine.processing.coin_mapper import normalize_symbol

logger = logging.getLogger(__name__)

# Cashtag/keyword patterns per symbol (matched against lowercased tweet text).
_SYMBOL_PATTERNS = {
    "BTC/USDT": re.compile(r"\$btc\b|\bbtc\b|\bbitcoin\b", re.IGNORECASE),
    "ADA/USDT": re.compile(r"\$ada\b|\bada\b|\bcardano\b", re.IGNORECASE),
    "ETH/USDT": re.compile(r"\$eth\b|\beth\b|\bethereum\b", re.IGNORECASE),
    "BNB/USDT": re.compile(r"\$bnb\b|\bbnb\b|\bbinance coin\b", re.IGNORECASE),
    "SOL/USDT": re.compile(r"\$sol\b|\bsol\b|\bsolana\b", re.IGNORECASE),
    "XRP/USDT": re.compile(r"\$xrp\b|\bxrp\b|\bripple\b", re.IGNORECASE),
    "DOGE/USDT": re.compile(r"\$doge\b|\bdoge\b|\bdogecoin\b", re.IGNORECASE),
    "LINK/USDT": re.compile(r"\$link\b|\bchainlink\b", re.IGNORECASE),
}
_CASHTAGS = {
    "BTC/USDT": "$BTC", "ADA/USDT": "$ADA", "ETH/USDT": "$ETH", "BNB/USDT": "$BNB",
    "SOL/USDT": "$SOL", "XRP/USDT": "$XRP", "DOGE/USDT": "$DOGE", "LINK/USDT": "$LINK",
}


def build_query(symbols: Sequence[str]) -> str:
    """Recent-search query: cashtags, no retweets, English."""
    tags = [_CASHTAGS[s] for s in symbols if s in _CASHTAGS]
    return f"({' OR '.join(tags)}) -is:retweet lang:en"


def parse_created_at(raw: Optional[str]) -> float:
    if not raw:
        return time.time()
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


def match_symbols(text: str, symbols: Sequence[str]) -> List[str]:
    return [s for s in symbols if s in _SYMBOL_PATTERNS and _SYMBOL_PATTERNS[s].search(text)]


class XFeed:
    """Budget-capped poller for X cashtag search."""

    def __init__(
        self,
        symbols: Sequence[str],
        engine_url: str,
        bearer_token: str = X_BEARER_TOKEN,
        poll_seconds: float = X_POLL_SECONDS,
        max_results: int = X_MAX_RESULTS,
        daily_read_budget: int = X_DAILY_READ_BUDGET,
    ) -> None:
        if not bearer_token:
            raise ValueError(
                "X feed requires SENTIMENT_X_BEARER_TOKEN. "
                "Create a key at console.x.com (pay-per-use) and set the env var."
            )
        self.symbols = [normalize_symbol(s) for s in symbols]
        self.engine_url = engine_url.rstrip("/")
        self.bearer_token = bearer_token
        self.poll_seconds = max(60.0, poll_seconds)
        self.max_results = max(10, min(100, max_results))  # API minimum is 10
        self.daily_read_budget = daily_read_budget
        self.query = build_query(self.symbols)
        self._since_id: Optional[str] = None
        self._budget_day: Optional[str] = None
        self._reads_today = 0
        self.pushed_count = 0

    # --- budget ------------------------------------------------------------

    def _budget_remaining(self, now: Optional[float] = None) -> int:
        day = time.strftime("%Y-%m-%d", time.gmtime(now if now is not None else time.time()))
        if day != self._budget_day:
            self._budget_day = day
            self._reads_today = 0
        return max(0, self.daily_read_budget - self._reads_today)

    def _consume_budget(self, reads: int) -> None:
        self._budget_remaining()  # roll the day first so reads land on today
        self._reads_today += reads

    # --- parsing (pure, testable) -------------------------------------------

    def process_response(self, data: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        """API payload -> [(symbol, text, created_ts)]; updates since_id/budget."""
        tweets = data.get("data") or []
        if not tweets:
            return []
        self._consume_budget(len(tweets))
        newest = data.get("meta", {}).get("newest_id")
        if newest:
            self._since_id = newest
        pushes: List[Tuple[str, str, float]] = []
        for tweet in tweets:
            text = str(tweet.get("text") or "")
            if not text:
                continue
            created = parse_created_at(tweet.get("created_at"))
            for symbol in match_symbols(text, self.symbols):
                pushes.append((symbol, text, created))
        return pushes

    # --- network loop ---------------------------------------------------------

    async def run(self) -> None:
        import httpx  # local import keeps unit tests dependency-free

        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        logger.info(
            "x feed: query=%r every %.0fs, daily budget %d reads (~$%.2f/day)",
            self.query, self.poll_seconds, self.daily_read_budget,
            self.daily_read_budget * 0.005,
        )
        backoff = 1.0
        async with httpx.AsyncClient(timeout=15.0) as client:
            while True:
                remaining = self._budget_remaining()
                if remaining <= 0:
                    logger.info("x feed: daily read budget exhausted; sleeping until tomorrow")
                    await asyncio.sleep(1800)
                    continue
                params: Dict[str, Any] = {
                    "query": self.query,
                    "max_results": min(self.max_results, max(10, remaining)),
                    "tweet.fields": "created_at",
                }
                if self._since_id:
                    params["since_id"] = self._since_id
                try:
                    response = await client.get(X_SEARCH_URL, params=params, headers=headers)
                    if response.status_code == 429:
                        backoff = min(backoff * 2, 3600)
                        logger.warning("x feed: rate limited; backing off %.0fs", backoff)
                        await asyncio.sleep(backoff)
                        continue
                    response.raise_for_status()
                    backoff = 1.0
                    pushes = self.process_response(response.json())
                except Exception as exc:  # noqa: BLE001 - feed must not die
                    logger.warning("x feed: poll failed (%s: %s)", type(exc).__name__, exc)
                    await asyncio.sleep(self.poll_seconds)
                    continue
                for symbol, text, created in pushes:
                    try:
                        await client.post(
                            f"{self.engine_url}/ingest/social",
                            json={"symbol": symbol, "source": "x:search",
                                  "text": text, "timestamp": created},
                        )
                        self.pushed_count += 1
                        logger.info("x feed: %s <- %r", symbol, text[:70])
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("x feed: push failed (%s)", type(exc).__name__)
                await asyncio.sleep(self.poll_seconds)

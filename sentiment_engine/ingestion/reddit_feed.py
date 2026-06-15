"""Reddit connector (keyless public JSON, read-only).

Polls /new.json for a few crypto subreddits, matches post titles to supported
symbols, and pushes them through /ingest/social (source "reddit:<sub>").
Same baseline-priming rule as the news feed: the existing backlog on startup
is marked seen without being pushed, so restarts cannot fake attention spikes.

Reddit's public JSON endpoints allow modest, well-behaved polling with a
descriptive User-Agent. 429 responses back off automatically.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Sequence, Set, Tuple

from sentiment_engine.config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_POLL_SECONDS,
    REDDIT_SUBREDDITS,
)
from sentiment_engine.processing.coin_mapper import aliases_for, normalize_symbol

logger = logging.getLogger(__name__)

_MAX_SEEN_IDS = 5000
_USER_AGENT = "sentiment-engine/0.1 (standalone research tool)"

# Posts in a coin's own subreddit are about that coin even when the title
# doesn't name it. Generic subs still require a title match.
IMPLIED_SYMBOL = {
    "bitcoin": "BTC/USDT", "btc": "BTC/USDT",
    "cardano": "ADA/USDT",
    "ethereum": "ETH/USDT", "ethfinance": "ETH/USDT",
    "binance": "BNB/USDT",
    "solana": "SOL/USDT",
    "ripple": "XRP/USDT", "xrp": "XRP/USDT",
    "dogecoin": "DOGE/USDT",
    "chainlink": "LINK/USDT",
}


def parse_listing(data: Dict[str, Any]) -> List[Tuple[str, str, float]]:
    """Reddit listing JSON -> [(post_id, title, created_utc)]."""
    posts: List[Tuple[str, str, float]] = []
    try:
        for child in data["data"]["children"]:
            post = child["data"]
            posts.append((str(post["name"]), str(post["title"]), float(post.get("created_utc") or 0)))
    except (KeyError, TypeError, ValueError):
        return []
    return posts


class RedditFeed:
    def __init__(
        self,
        symbols: Sequence[str],
        engine_url: str,
        subreddits: Sequence[str] = REDDIT_SUBREDDITS,
        poll_seconds: float = REDDIT_POLL_SECONDS,
        client_id: str = REDDIT_CLIENT_ID,
        client_secret: str = REDDIT_CLIENT_SECRET,
    ) -> None:
        self.symbols = [normalize_symbol(s) for s in symbols]
        self.engine_url = engine_url.rstrip("/")
        self.subreddits = list(subreddits)
        self.poll_seconds = poll_seconds
        self.client_id = client_id
        self.client_secret = client_secret
        self._token: str = ""
        self._token_expiry: float = 0.0
        self._seen: Set[str] = set()
        self._primed: Dict[str, bool] = {sub: False for sub in self.subreddits}
        self._patterns = [
            (symbol, re.compile(r"\b(" + "|".join(map(re.escape, aliases_for(symbol))) + r")\b", re.IGNORECASE))
            for symbol in self.symbols
        ]
        self.pushed_count = 0

    @property
    def use_oauth(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def listing_url(self, subreddit: str) -> str:
        host = "oauth.reddit.com" if self.use_oauth else "www.reddit.com"
        return f"https://{host}/r/{subreddit}/new.json?limit=50"

    def match_symbols(self, text: str) -> List[str]:
        return [symbol for symbol, pattern in self._patterns if pattern.search(text)]

    async def _ensure_token(self, client) -> bool:
        """Fetch/refresh a userless OAuth bearer token. Returns True if valid."""
        if not self.use_oauth:
            return False
        if self._token and time.time() < self._token_expiry - 60:
            return True
        try:
            resp = await client.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=(self.client_id, self.client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": _USER_AGENT},
            )
            resp.raise_for_status()
            payload = resp.json()
            self._token = str(payload.get("access_token") or "")
            self._token_expiry = time.time() + float(payload.get("expires_in") or 3600)
            if self._token:
                logger.info("reddit feed: OAuth token acquired (userless)")
                return True
            logger.error("reddit feed: OAuth response had no access_token")
        except Exception as exc:  # noqa: BLE001
            status = getattr(getattr(exc, "response", None), "status_code", None)
            logger.error("reddit feed: OAuth token request failed (%s status=%s)", type(exc).__name__, status)
        return False

    def process_listing(self, subreddit: str, data: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Returns (symbol, title) pairs to push. Pure; no network."""
        posts = parse_listing(data)
        if not posts:
            return []
        first_poll = not self._primed.get(subreddit, False)
        to_push: List[Tuple[str, str]] = []
        for post_id, title, _created in posts:
            if post_id in self._seen:
                continue
            self._seen.add(post_id)
            if first_poll:
                continue
            matched = self.match_symbols(title)
            if not matched:
                implied = IMPLIED_SYMBOL.get(subreddit.lower())
                if implied is not None and implied in self.symbols:
                    matched = [implied]
            for symbol in matched:
                to_push.append((symbol, title))
        self._primed[subreddit] = True
        if len(self._seen) > _MAX_SEEN_IDS:
            self._seen = set(list(self._seen)[-_MAX_SEEN_IDS // 2:])
        return to_push

    async def run(self) -> None:
        import httpx  # local import keeps unit tests dependency-free

        backoff = 1.0
        mode = "OAuth" if self.use_oauth else "keyless (no app id/secret set — unreliable in 2026; see README)"
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            logger.info("reddit feed: polling r/%s every %.0fs [%s]", ", r/".join(self.subreddits), self.poll_seconds, mode)
            while True:
                if self.use_oauth and not await self._ensure_token(client):
                    logger.warning("reddit feed: no OAuth token; retrying in %.0fs", self.poll_seconds)
                    await asyncio.sleep(self.poll_seconds)
                    continue
                for subreddit in self.subreddits:
                    headers = {"User-Agent": _USER_AGENT}
                    if self.use_oauth:
                        headers["Authorization"] = f"bearer {self._token}"
                    try:
                        response = await client.get(self.listing_url(subreddit), headers=headers)
                        if response.status_code in (429, 403):
                            backoff = min(backoff * 2, 600)
                            logger.warning(
                                "reddit feed: r/%s blocked (HTTP %s)%s; backing off %.0fs",
                                subreddit, response.status_code,
                                "" if self.use_oauth else " — set SENTIMENT_REDDIT_CLIENT_ID/SECRET for reliable access",
                                backoff,
                            )
                            await asyncio.sleep(backoff)
                            continue
                        response.raise_for_status()
                        backoff = 1.0
                        pushes = self.process_listing(subreddit, response.json())
                    except Exception as exc:  # noqa: BLE001
                        status = getattr(getattr(exc, "response", None), "status_code", None)
                        logger.warning("reddit feed: r/%s failed (%s status=%s)", subreddit, type(exc).__name__, status)
                        continue
                    for symbol, title in pushes:
                        try:
                            await client.post(
                                f"{self.engine_url}/ingest/social",
                                json={
                                    "symbol": symbol,
                                    "source": f"reddit:{subreddit}",
                                    "text": title,
                                    "timestamp": time.time(),
                                },
                            )
                            self.pushed_count += 1
                            logger.info("reddit feed: %s <- r/%s %r", symbol, subreddit, title[:70])
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("reddit feed: push failed (%s)", type(exc).__name__)
                await asyncio.sleep(self.poll_seconds)

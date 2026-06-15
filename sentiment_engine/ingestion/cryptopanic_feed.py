"""CryptoPanic connector (free news aggregator + crowd sentiment votes).

CryptoPanic aggregates crypto news across many outlets and lets its community
vote each item bullish/bearish. We poll the public posts API for our coins, map
each post to symbols via its `currencies`, derive a sentiment from the crowd
votes when present (otherwise the headline is scored by the engine), and push it
through /ingest/social (source "cryptopanic").

Dormant unless SENTIMENT_CRYPTOPANIC_TOKEN is set (a free token from
cryptopanic.com/developers). Same baseline-priming rule as the news feed: the
backlog on the first poll is marked seen without being pushed.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from sentiment_engine.config import (
    CRYPTOPANIC_POLL_SECONDS,
    CRYPTOPANIC_TOKEN,
    CRYPTOPANIC_URL,
    SUPPORTED_SYMBOLS,
)
from sentiment_engine.processing.coin_mapper import normalize_symbol

logger = logging.getLogger(__name__)

_MAX_SEEN_IDS = 5000

# (symbol, title, vote_sentiment-or-None)
CryptoPanicItem = Tuple[str, str, Optional[float]]


def vote_sentiment(votes: Optional[Dict[str, Any]]) -> Optional[float]:
    """Crowd votes -> [-1, 1] sentiment, or None if no directional votes."""
    if not isinstance(votes, dict):
        return None
    pos = float(votes.get("positive") or 0)
    neg = float(votes.get("negative") or 0)
    total = pos + neg
    if total <= 0:
        return None
    return max(-1.0, min(1.0, (pos - neg) / total))


def supported_codes() -> Dict[str, str]:
    """Map a bare currency code (BTC) -> supported symbol (BTC/USDT)."""
    out: Dict[str, str] = {}
    for sym in SUPPORTED_SYMBOLS:
        out[sym.split("/")[0].upper()] = sym
    return out


def parse_posts(data: Dict[str, Any], code_map: Dict[str, str]) -> List[Tuple[str, CryptoPanicItem]]:
    """CryptoPanic API JSON -> [(post_id, (symbol, title, vote_sentiment))]. Pure."""
    items: List[Tuple[str, CryptoPanicItem]] = []
    try:
        results = data["results"]
    except (KeyError, TypeError):
        return []
    for post in results:
        try:
            post_id = str(post["id"])
            title = str(post.get("title") or "").strip()
            if not title:
                continue
            sentiment = vote_sentiment(post.get("votes"))
            seen_syms: Set[str] = set()
            for cur in post.get("currencies") or []:
                code = str(cur.get("code") or "").upper()
                symbol = code_map.get(code)
                if symbol and symbol not in seen_syms:
                    seen_syms.add(symbol)
                    items.append((post_id, (symbol, title, sentiment)))
        except (KeyError, TypeError, ValueError):
            continue
    return items


class CryptoPanicFeed:
    def __init__(
        self,
        symbols: Sequence[str],
        engine_url: str,
        token: str = CRYPTOPANIC_TOKEN,
        url: str = CRYPTOPANIC_URL,
        poll_seconds: float = CRYPTOPANIC_POLL_SECONDS,
    ) -> None:
        self.symbols = [normalize_symbol(s) for s in symbols]
        self.engine_url = engine_url.rstrip("/")
        self.token = token
        self.url = url
        self.poll_seconds = poll_seconds
        self.code_map = {c: s for c, s in supported_codes().items() if s in self.symbols}
        self._seen: Set[str] = set()
        self._primed = False
        self.pushed_count = 0

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def process(self, data: Dict[str, Any]) -> List[CryptoPanicItem]:
        """Returns items to push (new since last poll). Pure; no network."""
        parsed = parse_posts(data, self.code_map)
        to_push: List[CryptoPanicItem] = []
        for post_id, item in parsed:
            key = f"{post_id}:{item[0]}"
            if key in self._seen:
                continue
            self._seen.add(key)
            if not self._primed:
                continue
            to_push.append(item)
        self._primed = True
        if len(self._seen) > _MAX_SEEN_IDS:
            self._seen = set(list(self._seen)[-_MAX_SEEN_IDS // 2:])
        return to_push

    async def run(self) -> None:
        import httpx

        if not self.enabled:
            logger.info("cryptopanic feed: disabled (set SENTIMENT_CRYPTOPANIC_TOKEN to enable)")
            return
        codes = ",".join(self.code_map.keys())
        params = {"auth_token": self.token, "public": "true", "currencies": codes}
        backoff = 1.0
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            logger.info("cryptopanic feed: polling %s every %.0fs", codes, self.poll_seconds)
            while True:
                try:
                    resp = await client.get(self.url, params=params, headers={"User-Agent": "sentiment-engine/0.1"})
                    if resp.status_code == 429:
                        backoff = min(backoff * 2, 600)
                        logger.warning("cryptopanic feed: rate limited; backing off %.0fs", backoff)
                        await asyncio.sleep(backoff)
                        continue
                    resp.raise_for_status()
                    backoff = 1.0
                    pushes = self.process(resp.json())
                except Exception as exc:  # noqa: BLE001
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    logger.warning("cryptopanic feed: fetch failed (%s status=%s)", type(exc).__name__, status)
                    await asyncio.sleep(self.poll_seconds)
                    continue
                for symbol, title, sentiment in pushes:
                    body: Dict[str, Any] = {
                        "symbol": symbol, "source": "cryptopanic", "text": title, "timestamp": time.time(),
                    }
                    if sentiment is not None:
                        body["sentiment"] = sentiment
                    try:
                        await client.post(f"{self.engine_url}/ingest/social", json=body)
                        self.pushed_count += 1
                        logger.info("cryptopanic feed: %s <- %r (vote=%s)", symbol, title[:60], sentiment)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("cryptopanic feed: push failed (%s)", type(exc).__name__)
                await asyncio.sleep(self.poll_seconds)

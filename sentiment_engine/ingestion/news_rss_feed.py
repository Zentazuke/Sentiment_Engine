"""News RSS connector (keyless, read-only).

Polls a few crypto news RSS/Atom feeds, matches headlines to supported
symbols, and pushes them through the engine's existing /ingest/social
pipeline (source "news:<domain>"). The engine's keyword scorer and velocity
logic do the rest; a single headline can never flip a decision because
sentiment only nudges confidence.

Shock-flood protection: on the FIRST successful poll of each feed, existing
items are marked seen WITHOUT being pushed. Otherwise a restart would inject
the whole backlog at once and fake an attention spike. Only items appearing
after startup are pushed.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

from sentiment_engine.config import NEWS_IMPLIED_SYMBOLS, NEWS_POLL_SECONDS, NEWS_RSS_URLS
from sentiment_engine.processing.coin_mapper import aliases_for, normalize_symbol

logger = logging.getLogger(__name__)

_MAX_SEEN_IDS = 5000
_ATOM_NS = "{http://www.w3.org/2005/Atom}"


@dataclass(frozen=True, slots=True)
class NewsItem:
    item_id: str
    title: str
    published_at: Optional[float]  # unix seconds, None if unparseable


def _parse_pubdate(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw.strip()).timestamp()
    except (TypeError, ValueError):
        return None


def parse_feed(xml_text: str) -> List[NewsItem]:
    """Parse RSS 2.0 or Atom into NewsItems. Unparseable feeds return []."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("news feed: XML parse error (%s)", exc)
        return []

    items: List[NewsItem] = []
    # RSS 2.0: <rss><channel><item>
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        item_id = (item.findtext("guid") or item.findtext("link") or title).strip()
        if title and item_id:
            items.append(NewsItem(item_id, title, _parse_pubdate(item.findtext("pubDate"))))
    # Atom: <feed><entry>
    for entry in root.iter(f"{_ATOM_NS}entry"):
        title = (entry.findtext(f"{_ATOM_NS}title") or "").strip()
        item_id = (entry.findtext(f"{_ATOM_NS}id") or title).strip()
        published = entry.findtext(f"{_ATOM_NS}published") or entry.findtext(f"{_ATOM_NS}updated")
        published_ts: Optional[float] = None
        if published:
            try:
                from datetime import datetime

                published_ts = datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp()
            except ValueError:
                published_ts = None
        if title and item_id:
            items.append(NewsItem(item_id, title, published_ts))
    return items


class NewsRssFeed:
    """Polls RSS feeds and forwards symbol-matched headlines to the engine."""

    def __init__(
        self,
        symbols: Sequence[str],
        engine_url: str,
        urls: Sequence[str] = NEWS_RSS_URLS,
        poll_seconds: float = NEWS_POLL_SECONDS,
        implied_symbols: Optional[Dict[str, str]] = None,
    ) -> None:
        self.symbols = [normalize_symbol(s) for s in symbols]
        self.engine_url = engine_url
        self.urls = list(urls)
        self.poll_seconds = poll_seconds
        self.implied_symbols = dict(NEWS_IMPLIED_SYMBOLS if implied_symbols is None else implied_symbols)
        self._seen: Set[str] = set()
        self._primed: Dict[str, bool] = {url: False for url in self.urls}
        self._patterns = [
            (symbol, re.compile(r"\b(" + "|".join(map(re.escape, aliases_for(symbol))) + r")\b", re.IGNORECASE))
            for symbol in self.symbols
        ]
        self.pushed_count = 0

    def match_symbols(self, text: str) -> List[str]:
        return [symbol for symbol, pattern in self._patterns if pattern.search(text)]

    def symbols_for(self, url: str, title: str) -> List[str]:
        """Headline matches, falling back to the feed's implied coin if any."""
        matched = self.match_symbols(title)
        if not matched:
            implied = self.implied_symbols.get(url)
            if implied is not None and implied in self.symbols:
                matched = [implied]
        return matched

    def process_feed_text(self, url: str, xml_text: str) -> List[Tuple[str, str]]:
        """Returns (symbol, headline) pairs to push. Pure; no network."""
        items = parse_feed(xml_text)
        if not items:
            return []
        first_poll = not self._primed.get(url, False)
        to_push: List[Tuple[str, str]] = []
        for item in items:
            if item.item_id in self._seen:
                continue
            self._seen.add(item.item_id)
            if first_poll:
                continue  # baseline: never flood the backlog on startup
            for symbol in self.symbols_for(url, item.title):
                to_push.append((symbol, item.title))
        self._primed[url] = True
        if len(self._seen) > _MAX_SEEN_IDS:
            # Crude cap; ids of long-gone items are safe to forget.
            self._seen = set(list(self._seen)[-_MAX_SEEN_IDS // 2:])
        return to_push

    async def run(self) -> None:
        """Poll feeds forever. All failures are logged and non-fatal."""
        import httpx  # local import keeps unit tests dependency-free

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            logger.info("news feed: polling %d sources every %.0fs", len(self.urls), self.poll_seconds)
            while True:
                for url in self.urls:
                    try:
                        response = await client.get(url, headers={"User-Agent": "sentiment-engine/0.1"})
                        response.raise_for_status()
                        pushes = self.process_feed_text(url, response.text)
                    except httpx.HTTPError as exc:
                        logger.warning("news feed: fetch failed %s (%s)", url, type(exc).__name__)
                        continue
                    for symbol, title in pushes:
                        try:
                            await client.post(
                                f"{self.engine_url}/ingest/social",
                                json={
                                    "symbol": symbol,
                                    "source": f"news:{urlparse(url).netloc}",
                                    "text": title,
                                    "timestamp": time.time(),
                                },
                            )
                            self.pushed_count += 1
                            logger.info("news feed: %s <- %r", symbol, title[:80])
                        except httpx.HTTPError as exc:
                            logger.warning("news feed: push failed (%s)", type(exc).__name__)
                await asyncio.sleep(self.poll_seconds)

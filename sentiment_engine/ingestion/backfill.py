"""One-shot history backfill from news RSS and Reddit (keyless, read-only).

RSS items carry publish dates and Reddit posts carry created_utc, typically
covering the last 1-3 days. This command ingests that backlog with its REAL
timestamps so the 6h/24h outlook windows are populated immediately instead of
waiting a day for live accumulation.

Safe to re-run: the engine's social history has a uniqueness guard, so
duplicate items are ignored.

Run:
    python -m sentiment_engine.ingestion.backfill --sources news reddit --symbols BTC/USDT ADA/USDT --hours 48
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import List, Optional, Tuple

from sentiment_engine.config import DEFAULT_ENGINE_URL, NEWS_RSS_URLS, REDDIT_SUBREDDITS, SUPPORTED_SYMBOLS
from sentiment_engine.ingestion.news_rss_feed import NewsRssFeed, parse_feed
from sentiment_engine.ingestion.reddit_feed import IMPLIED_SYMBOL, RedditFeed, parse_listing
from sentiment_engine.processing.coin_mapper import normalize_symbol
from urllib.parse import urlparse

logger = logging.getLogger("sentiment_engine.backfill")

# (symbol, source, text, timestamp)
BackfillItem = Tuple[str, str, str, float]


def collect_news_items(matcher: NewsRssFeed, url: str, xml_text: str, now: float, hours: float) -> List[BackfillItem]:
    """Pure: RSS xml -> backfill items inside the window, with real timestamps."""
    cutoff = now - hours * 3600.0
    items: List[BackfillItem] = []
    for item in parse_feed(xml_text):
        if item.published_at is None or not (cutoff <= item.published_at <= now + 60):
            continue
        for symbol in matcher.symbols_for(url, item.title):
            items.append((symbol, f"news:{urlparse(url).netloc}", item.title, item.published_at))
    return items


def collect_reddit_items(matcher: RedditFeed, subreddit: str, data: dict, now: float, hours: float) -> List[BackfillItem]:
    """Pure: Reddit listing -> backfill items inside the window."""
    cutoff = now - hours * 3600.0
    items: List[BackfillItem] = []
    for _post_id, title, created in parse_listing(data):
        if not (cutoff <= created <= now + 60):
            continue
        matched = matcher.match_symbols(title)
        if not matched:
            implied = IMPLIED_SYMBOL.get(subreddit.lower())
            if implied is not None and implied in matcher.symbols:
                matched = [implied]
        for symbol in matched:
            items.append((symbol, f"reddit:{subreddit}", title, created))
    return items


async def run_backfill(sources: List[str], symbols: List[str], engine_url: str, hours: float) -> int:
    import httpx

    now = time.time()
    pushed = 0
    news_matcher = NewsRssFeed(symbols, engine_url=engine_url)
    reddit_matcher = RedditFeed(symbols, engine_url=engine_url)
    collected: List[BackfillItem] = []

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        if "news" in sources:
            for url in NEWS_RSS_URLS:
                try:
                    response = await client.get(url, headers={"User-Agent": "sentiment-engine/0.1"})
                    response.raise_for_status()
                    items = collect_news_items(news_matcher, url, response.text, now, hours)
                    collected.extend(items)
                    logger.info("news %s: %d matched items", urlparse(url).netloc, len(items))
                except Exception as exc:  # noqa: BLE001 - best effort per feed
                    logger.warning("news %s failed (%s)", url, type(exc).__name__)
        if "reddit" in sources:
            authed = await reddit_matcher._ensure_token(client)  # uses OAuth if creds set
            for subreddit in REDDIT_SUBREDDITS:
                headers = {"User-Agent": "sentiment-engine/0.1 (backfill)"}
                if authed:
                    headers["Authorization"] = f"bearer {reddit_matcher._token}"
                host = "oauth.reddit.com" if authed else "www.reddit.com"
                try:
                    response = await client.get(
                        f"https://{host}/r/{subreddit}/new.json?limit=100",
                        headers=headers,
                    )
                    if response.status_code in (403, 429) and not authed:
                        logger.warning(
                            "r/%s blocked (HTTP %s) — set SENTIMENT_REDDIT_CLIENT_ID/SECRET for reliable backfill",
                            subreddit, response.status_code,
                        )
                        continue
                    response.raise_for_status()
                    items = collect_reddit_items(reddit_matcher, subreddit, response.json(), now, hours)
                    collected.extend(items)
                    logger.info("r/%s: %d matched items", subreddit, len(items))
                except Exception as exc:  # noqa: BLE001
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    logger.warning("r/%s failed (%s status=%s)", subreddit, type(exc).__name__, status)

        collected.sort(key=lambda item: item[3])  # oldest first, natural order
        for symbol, source, text, timestamp in collected:
            try:
                response = await client.post(
                    f"{engine_url.rstrip('/')}/ingest/social",
                    json={"symbol": symbol, "source": source, "text": text, "timestamp": timestamp},
                )
                response.raise_for_status()
                pushed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("push failed (%s): %r", type(exc).__name__, text[:60])
    return pushed


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="One-shot social/news history backfill.")
    parser.add_argument("--sources", nargs="+", default=["news", "reddit"], choices=["news", "reddit"])
    parser.add_argument("--symbols", nargs="+", default=list(SUPPORTED_SYMBOLS))
    parser.add_argument("--engine-url", default=DEFAULT_ENGINE_URL)
    parser.add_argument("--hours", type=float, default=48.0)
    args = parser.parse_args()
    symbols = [normalize_symbol(symbol) for symbol in args.symbols]
    pushed = asyncio.run(run_backfill(args.sources, symbols, args.engine_url, args.hours))
    logger.info("backfill complete: %d items pushed (duplicates ignored engine-side)", pushed)


if __name__ == "__main__":
    main()

"""Live market ingestion runner.

Runs the Binance trade + order book feeds and pushes computed microstructure
metrics into the standalone sentiment engine over HTTP. The engine stays a
separate process; if this runner dies, engine metrics simply go stale and the
engine leans neutral.

Run:
    python -m sentiment_engine.ingestion.run_live --sources binance --symbols BTC/USDT ADA/USDT

This process is read-only with respect to markets: it never places trades,
never authenticates with the exchange, and never generates trade signals.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from typing import List, Optional, Sequence

import httpx

from sentiment_engine.config import (
    DEFAULT_ENGINE_URL,
    JOURNAL_DB_PATH,
    MICRO_PUSH_INTERVAL_SECONDS,
    SNAPSHOT_DB_PATH,
    SNAPSHOT_INTERVAL_SECONDS,
    SUPPORTED_SYMBOLS,
)
from sentiment_engine.ingestion.binance_orderbook_feed import BinanceOrderBookFeed
from sentiment_engine.ingestion.binance_lsr_feed import BinanceLsrFeed
from sentiment_engine.ingestion.binance_derivatives_feed import BinanceDerivativesFeed
from sentiment_engine.ingestion.bluesky_feed import BlueskyFeed
from sentiment_engine.ingestion.cryptopanic_feed import CryptoPanicFeed
from sentiment_engine.ingestion.context_feed import ContextFeed
from sentiment_engine.ingestion.news_rss_feed import NewsRssFeed
from sentiment_engine.ingestion.reddit_feed import RedditFeed
from sentiment_engine.ingestion.x_feed import XFeed
from sentiment_engine.ingestion.binance_trade_feed import BinanceTradeFeed
from sentiment_engine.processing.coin_mapper import normalize_symbol

logger = logging.getLogger("sentiment_engine.run_live")

_STATS_LOG_EVERY_SECONDS = 30.0


def build_payload(
    symbol: str,
    trade_feed: BinanceTradeFeed,
    book_feed: BinanceOrderBookFeed,
    now: Optional[float] = None,
) -> Optional[dict]:
    """Merge trade + book metrics into one /ingest/microstructure payload.

    Returns None when there is nothing at all to report yet.
    """
    current = now if now is not None else time.time()
    metrics = {**trade_feed.metrics(symbol, now=current), **book_feed.metrics(symbol, now=current)}
    if not any(value is not None for value in metrics.values()):
        return None
    payload = {key: value for key, value in metrics.items() if value is not None}
    payload["symbol"] = symbol
    payload["computed_at"] = current
    return payload


async def push_loop(
    symbols: Sequence[str],
    trade_feed: BinanceTradeFeed,
    book_feed: BinanceOrderBookFeed,
    engine_url: str,
    interval: float,
) -> None:
    """Periodically push metrics to the engine. Engine downtime is non-fatal."""
    last_stats = 0.0
    async with httpx.AsyncClient(base_url=engine_url, timeout=5.0) as client:
        while True:
            await asyncio.sleep(interval)
            now = time.time()
            for symbol in symbols:
                payload = build_payload(symbol, trade_feed, book_feed, now=now)
                if payload is None:
                    continue
                try:
                    response = await client.post("/ingest/microstructure", json=payload)
                    response.raise_for_status()
                    last_price = payload.get("last_price")
                    if last_price is not None:
                        await client.post(
                            "/ingest/price",
                            json={"symbol": symbol, "price": last_price, "timestamp": now},
                        )
                except httpx.HTTPError as exc:
                    logger.warning("push failed for %s (%s: %s) - engine offline?",
                                   symbol, type(exc).__name__, exc)
            if now - last_stats >= _STATS_LOG_EVERY_SECONDS:
                last_stats = now
                for symbol in symbols:
                    snapshot = build_payload(symbol, trade_feed, book_feed, now=now) or {}
                    logger.info(
                        "%s price=%s imb60=%s spread_bps=%s relvol=%s trades60=%s",
                        symbol,
                        snapshot.get("last_price"),
                        _fmt(snapshot.get("trade_imbalance_60s")),
                        _fmt(snapshot.get("spread_bps")),
                        _fmt(snapshot.get("relative_volume")),
                        snapshot.get("trade_count_60s"),
                    )


def _fmt(value: Optional[float]) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


async def main_async(args: argparse.Namespace) -> None:
    symbols = [normalize_symbol(symbol) for symbol in args.symbols]
    logger.info("starting live ingestion: sources=%s symbols=%s engine=%s",
                args.sources, symbols, args.engine_url)
    trade_feed = BinanceTradeFeed(symbols)
    book_feed = BinanceOrderBookFeed(symbols)
    tasks = [
        asyncio.create_task(trade_feed.run(), name="binance-trades"),
        asyncio.create_task(book_feed.run(), name="binance-book"),
        asyncio.create_task(
            push_loop(symbols, trade_feed, book_feed, args.engine_url, args.push_interval),
            name="push-loop",
        ),
    ]
    if "news" in args.sources:
        news_feed = NewsRssFeed(symbols, engine_url=args.engine_url)
        tasks.append(asyncio.create_task(news_feed.run(), name="news-rss"))
    if "reddit" in args.sources:
        reddit_feed = RedditFeed(symbols, engine_url=args.engine_url)
        tasks.append(asyncio.create_task(reddit_feed.run(), name="reddit"))
    if "bluesky" in args.sources:
        bluesky_feed = BlueskyFeed(symbols, engine_url=args.engine_url)
        tasks.append(asyncio.create_task(bluesky_feed.run(), name="bluesky"))
    if "cryptopanic" in args.sources:
        cryptopanic_feed = CryptoPanicFeed(symbols, engine_url=args.engine_url)
        tasks.append(asyncio.create_task(cryptopanic_feed.run(), name="cryptopanic"))
    if "context" in args.sources:
        context_feed = ContextFeed(engine_url=args.engine_url)
        tasks.append(asyncio.create_task(context_feed.run(), name="context"))
    if "lsr" in args.sources:
        lsr_feed = BinanceLsrFeed(symbols, engine_url=args.engine_url)
        tasks.append(asyncio.create_task(lsr_feed.run(), name="lsr"))
    if "deriv" in args.sources:
        deriv_feed = BinanceDerivativesFeed(symbols, engine_url=args.engine_url)
        tasks.append(asyncio.create_task(deriv_feed.run(), name="deriv"))
    if not getattr(args, "no_snapshot", False):
        from sentiment_engine.storage.snapshot import snapshot_loop

        tasks.append(asyncio.create_task(
            snapshot_loop(JOURNAL_DB_PATH, SNAPSHOT_DB_PATH, SNAPSHOT_INTERVAL_SECONDS),
            name="db-snapshot",
        ))
    if "x" in args.sources:
        try:
            x_feed = XFeed(symbols, engine_url=args.engine_url)
            tasks.append(asyncio.create_task(x_feed.run(), name="x-search"))
        except ValueError as exc:
            logger.warning("x feed disabled: %s", exc)
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        raise


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live market ingestion for the sentiment engine.")
    parser.add_argument("--sources", nargs="+", default=["binance"],
                        help="Data sources: binance (required), news, reddit, bluesky, context, lsr, deriv, x (needs SENTIMENT_X_BEARER_TOKEN).")
    parser.add_argument("--symbols", nargs="+", default=list(SUPPORTED_SYMBOLS),
                        help="Symbols like BTC/USDT ADA/USDT.")
    parser.add_argument("--engine-url", default=DEFAULT_ENGINE_URL,
                        help="Base URL of the sentiment engine API.")
    parser.add_argument("--push-interval", type=float, default=MICRO_PUSH_INTERVAL_SECONDS,
                        help="Seconds between metric pushes.")
    parser.add_argument("--no-snapshot", action="store_true",
                        help="Disable periodic DB snapshot into the project folder.")
    args = parser.parse_args(argv)
    supported = {"binance", "news", "reddit", "bluesky", "cryptopanic", "context", "lsr", "deriv", "x"}
    unsupported = [source for source in args.sources if source not in supported]
    if unsupported:
        parser.error(f"Unsupported sources: {unsupported}. Supported: {sorted(supported)}.")
    if "binance" not in args.sources:
        parser.error("'binance' must be included: microstructure is the primary layer.")
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        logger.info("stopped by user")


if __name__ == "__main__":
    main()

"""Run on your machine to verify each news feed and per-coin coverage.

    python diagnose_news.py

Fetches every configured feed, parses it, and reports HTTP status plus how
many items match BTC vs ADA (including implied-coin feeds). The point is to
confirm ADA is no longer sentiment-blind.
"""
import asyncio

from sentiment_engine.config import NEWS_RSS_URLS
from sentiment_engine.ingestion.news_rss_feed import NewsRssFeed, parse_feed


async def main():
    import httpx

    matcher = NewsRssFeed(["BTC/USDT", "ADA/USDT"], engine_url="http://x")
    totals = {"BTC/USDT": 0, "ADA/USDT": 0}
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as c:
        for url in NEWS_RSS_URLS:
            try:
                r = await c.get(url, headers={"User-Agent": "sentiment-engine/0.1"})
                items = parse_feed(r.text) if r.status_code == 200 else []
                btc = ada = 0
                for it in items:
                    syms = matcher.symbols_for(url, it.title)
                    btc += "BTC/USDT" in syms
                    ada += "ADA/USDT" in syms
                totals["BTC/USDT"] += btc
                totals["ADA/USDT"] += ada
                tag = "  <- implied ADA" if "cardano" in url.lower() or "Cardano" in url else ""
                print(f"HTTP {r.status_code}  items={len(items):>3}  BTC={btc:>2} ADA={ada:>2}  {url[:60]}{tag}")
            except Exception as e:
                print(f"FAILED {type(e).__name__:20} {url[:60]}")
    print(f"\nTOTAL matched this poll:  BTC={totals['BTC/USDT']}  ADA={totals['ADA/USDT']}")
    if totals["ADA/USDT"] == 0:
        print("ADA still 0 -- the coin-targeted feeds may be unreachable from here; tell Claude.")


if __name__ == "__main__":
    asyncio.run(main())

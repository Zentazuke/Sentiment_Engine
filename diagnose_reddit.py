"""Run on YOUR machine to see exactly what Reddit returns from your IP.

    python diagnose_reddit.py

Tries the keyless endpoint first, then OAuth if SENTIMENT_REDDIT_CLIENT_ID /
SENTIMENT_REDDIT_CLIENT_SECRET are set. Prints HTTP status + how many posts
came back so we can tell blocked-by-Reddit from a code bug.
"""
import asyncio
import os

SUBS = os.getenv("SENTIMENT_REDDIT_SUBREDDITS", "CryptoCurrency,Bitcoin,cardano").split(",")
UA = "sentiment-engine/0.1 (diagnostic)"


async def main():
    import httpx

    cid = os.getenv("SENTIMENT_REDDIT_CLIENT_ID", "").strip()
    csec = os.getenv("SENTIMENT_REDDIT_CLIENT_SECRET", "").strip()
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as c:
        print("=== KEYLESS (www.reddit.com) ===")
        for sub in SUBS:
            try:
                r = await c.get(f"https://www.reddit.com/r/{sub.strip()}/new.json?limit=10",
                                headers={"User-Agent": UA})
                n = len(r.json().get("data", {}).get("children", [])) if r.status_code == 200 else 0
                print(f"  r/{sub.strip():16} HTTP {r.status_code}  posts={n}")
            except Exception as e:
                print(f"  r/{sub.strip():16} ERROR {type(e).__name__}")

        if not (cid and csec):
            print("\nNo OAuth creds set. To enable the reliable path:")
            print("  1. https://www.reddit.com/prefs/apps -> create app -> type 'script'")
            print("  2. set SENTIMENT_REDDIT_CLIENT_ID and SENTIMENT_REDDIT_CLIENT_SECRET")
            return

        print("\n=== OAUTH (oauth.reddit.com) ===")
        try:
            tok = await c.post("https://www.reddit.com/api/v1/access_token",
                               auth=(cid, csec), data={"grant_type": "client_credentials"},
                               headers={"User-Agent": UA})
            print(f"  token request HTTP {tok.status_code}")
            tok.raise_for_status()
            bearer = tok.json()["access_token"]
        except Exception as e:
            print(f"  token request FAILED {type(e).__name__} — check id/secret")
            return
        for sub in SUBS:
            try:
                r = await c.get(f"https://oauth.reddit.com/r/{sub.strip()}/new.json?limit=10",
                                headers={"User-Agent": UA, "Authorization": f"bearer {bearer}"})
                n = len(r.json().get("data", {}).get("children", [])) if r.status_code == 200 else 0
                print(f"  r/{sub.strip():16} HTTP {r.status_code}  posts={n}")
            except Exception as e:
                print(f"  r/{sub.strip():16} ERROR {type(e).__name__}")


if __name__ == "__main__":
    asyncio.run(main())

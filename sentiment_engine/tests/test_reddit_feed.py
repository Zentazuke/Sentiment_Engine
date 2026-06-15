"""Reddit feed OAuth / host-selection / diagnostics-logging tests."""

import asyncio

from sentiment_engine.ingestion.reddit_feed import RedditFeed


def _feed(**kw):
    return RedditFeed(["BTC/USDT", "ADA/USDT"], engine_url="http://x", **kw)


def test_keyless_by_default():
    f = _feed(client_id="", client_secret="")
    assert f.use_oauth is False
    assert "www.reddit.com" in f.listing_url("Bitcoin")


def test_oauth_when_creds_present():
    f = _feed(client_id="abc", client_secret="def")
    assert f.use_oauth is True
    assert "oauth.reddit.com" in f.listing_url("Bitcoin")


def test_oauth_requires_both_halves():
    assert _feed(client_id="abc", client_secret="").use_oauth is False
    assert _feed(client_id="", client_secret="def").use_oauth is False


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    """Records the access-token POST and returns a canned token."""

    def __init__(self):
        self.posted = None

    async def post(self, url, auth=None, data=None, headers=None):
        self.posted = {"url": url, "auth": auth, "data": data}
        return _FakeResp({"access_token": "TKN123", "expires_in": 3600})


def test_ensure_token_keyless_returns_false():
    f = _feed()
    client = _FakeClient()
    assert asyncio.run(f._ensure_token(client)) is False
    assert client.posted is None  # never hits the network without creds


def test_ensure_token_fetches_and_caches():
    f = _feed(client_id="id", client_secret="secret")
    client = _FakeClient()
    assert asyncio.run(f._ensure_token(client)) is True
    assert f._token == "TKN123"
    assert client.posted["auth"] == ("id", "secret")
    assert client.posted["data"]["grant_type"] == "client_credentials"
    # cached: a second call with a fresh client must not re-POST
    client2 = _FakeClient()
    assert asyncio.run(f._ensure_token(client2)) is True
    assert client2.posted is None


def test_process_listing_still_matches_implied_subreddit():
    f = _feed()
    listing = {"data": {"children": [
        {"data": {"name": "t3_1", "title": "random post with no ticker", "created_utc": 1.0}},
    ]}}
    f.process_listing("Bitcoin", listing)  # first poll primes (no push)
    listing2 = {"data": {"children": [
        {"data": {"name": "t3_2", "title": "another untagged post", "created_utc": 2.0}},
    ]}}
    pushes = f.process_listing("Bitcoin", listing2)
    assert ("BTC/USDT", "another untagged post") in pushes

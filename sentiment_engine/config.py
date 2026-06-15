"""Configuration for the standalone sentiment engine.

All values can be overridden with environment variables. The defaults are
conservative because this service should never force a trading decision.
"""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Final, Tuple


def _decimal_env(name: str, default: str) -> Decimal:
    raw = os.getenv(name, default).strip()
    try:
        return Decimal(raw)
    except Exception as exc:  # noqa: BLE001 - boot-time config guard
        raise RuntimeError(f"Invalid decimal config {name}={raw!r}") from exc


def _int_env(name: str, default: str) -> int:
    raw = os.getenv(name, default).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer config {name}={raw!r}") from exc


def _str_env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _tuple_env(name: str, default: str) -> Tuple[str, ...]:
    values = tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())
    if not values:
        raise RuntimeError(f"{name} must contain at least one symbol")
    return values


SERVICE_HOST: Final[str] = os.getenv("SENTIMENT_HOST", "127.0.0.1")
SERVICE_PORT: Final[int] = _int_env("SENTIMENT_PORT", "8787")
SUPPORTED_SYMBOLS: Final[Tuple[str, ...]] = _tuple_env(
    "SENTIMENT_SYMBOLS",
    "BTC/USDT,ADA/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT,DOGE/USDT,LINK/USDT"
)

# Main scoring window: the bot can query this snapshot whenever it has a setup.
MAIN_WINDOW_SECONDS: Final[int] = _int_env("SENTIMENT_WINDOW_SECONDS", "60")
PREVIOUS_WINDOW_SECONDS: Final[int] = _int_env("SENTIMENT_PREVIOUS_WINDOW_SECONDS", "60")
MAX_EVENT_AGE_SECONDS: Final[int] = MAIN_WINDOW_SECONDS + PREVIOUS_WINDOW_SECONDS + 30

# Data-quality / decision thresholds.
MIN_MESSAGES_FOR_SIGNAL: Final[int] = _int_env("SENTIMENT_MIN_MESSAGES_FOR_SIGNAL", "10")
MIN_DATA_QUALITY: Final[Decimal] = _decimal_env("SENTIMENT_MIN_DATA_QUALITY", "0.60")
ATTENTION_SPIKE_CONFIRM: Final[Decimal] = _decimal_env("SENTIMENT_ATTENTION_SPIKE_CONFIRM", "0.55")
SENTIMENT_VELOCITY_CONFIRM: Final[Decimal] = _decimal_env("SENTIMENT_VELOCITY_CONFIRM", "0.15")
SENTIMENT_VELOCITY_VETO: Final[Decimal] = _decimal_env("SENTIMENT_VELOCITY_VETO", "0.25")

# The standalone service returns an estimate, but phase 1 should not alter size.
MAX_CONFIDENCE_BOOST: Final[Decimal] = _decimal_env("SENTIMENT_MAX_CONFIDENCE_BOOST", "1.10")
MAX_CONFIDENCE_REDUCTION: Final[Decimal] = _decimal_env("SENTIMENT_MAX_CONFIDENCE_REDUCTION", "0.90")
NEUTRAL_CONFIDENCE_MODIFIER: Final[Decimal] = Decimal("1.00")

def _default_db_dir() -> str:
    """Local, NON-synced data dir. Running SQLite on a cloud-synced folder
    causes torn writes and DB corruption, so the live journal lives here."""
    base = os.getenv("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "sentiment_engine")


# Live journal: local disk by default (override with SENTIMENT_JOURNAL_DB).
JOURNAL_DB_PATH: Final[str] = os.getenv("SENTIMENT_JOURNAL_DB", os.path.join(_default_db_dir(), "sentiment_journal.db"))
# Periodic read-only snapshot written INTO the (synced) project folder so the
# data is reviewable without touching the live writer. Relative -> project cwd.
SNAPSHOT_DB_PATH: Final[str] = os.getenv("SENTIMENT_SNAPSHOT_DB", "sentiment_journal_snapshot.db")
SNAPSHOT_INTERVAL_SECONDS: Final[float] = float(os.getenv("SENTIMENT_SNAPSHOT_INTERVAL", "600"))
# Learned outlook calibration model (written by `python -m sentiment_engine.signals.calibrate`).
# Lives in the project folder; read at outlook time when adopted. Small JSON, safe to sync.
CALIBRATION_MODEL_PATH: Final[str] = os.getenv("SENTIMENT_CALIBRATION_MODEL", "calibration_model.json")

# --- Binance long/short ratio (keyless crowd-positioning feed) ---
LSR_BASE_URL: Final[str] = os.getenv("SENTIMENT_LSR_BASE_URL", "https://fapi.binance.com")
LSR_PERIOD: Final[str] = os.getenv("SENTIMENT_LSR_PERIOD", "5m")
LSR_POLL_SECONDS: Final[float] = float(os.getenv("SENTIMENT_LSR_POLL_SECONDS", "300"))
# Funding rate + open interest poller (same keyless host as the long/short feed).
DERIV_POLL_SECONDS: Final[float] = float(os.getenv("SENTIMENT_DERIV_POLL_SECONDS", "300"))
DEFAULT_ACTION_ON_ERROR: Final[str] = "neutral"


def _float_env(name: str, default: str) -> float:
    raw = os.getenv(name, default).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float config {name}={raw!r}") from exc


# --- Microstructure (live Binance feeds) ---
# A microstructure snapshot older than this is marked stale; stale metrics
# must never confirm or veto anything.
MICRO_STALE_SECONDS: Final[float] = _float_env("SENTIMENT_MICRO_STALE_SECONDS", "10")
# Depth band around mid price used for "liquidity near price", in percent.
MICRO_DEPTH_BAND_PCT: Final[float] = _float_env("SENTIMENT_MICRO_DEPTH_BAND_PCT", "0.25")
# Trailing buffer used as the relative-volume baseline, and the minimum
# buffer age before relative volume is reported at all (warmup).
MICRO_BASELINE_SECONDS: Final[float] = _float_env("SENTIMENT_MICRO_BASELINE_SECONDS", "900")
MICRO_BASELINE_MIN_SECONDS: Final[float] = _float_env("SENTIMENT_MICRO_BASELINE_MIN_SECONDS", "300")
# How often the feed runner pushes metrics into the engine.
MICRO_PUSH_INTERVAL_SECONDS: Final[float] = _float_env("SENTIMENT_MICRO_PUSH_INTERVAL", "1.0")
# VWAP window for vwap_distance_bps.
MICRO_VWAP_WINDOW_SECONDS: Final[float] = _float_env("SENTIMENT_MICRO_VWAP_WINDOW", "300")
BINANCE_WS_URL: Final[str] = os.getenv("SENTIMENT_BINANCE_WS_URL", "wss://stream.binance.com:9443/stream")
DEFAULT_ENGINE_URL: Final[str] = os.getenv("SENTIMENT_ENGINE_URL", "http://127.0.0.1:8787")

# --- Microstructure history (calibration recording) ---
MICRO_HISTORY_RETENTION_DAYS: Final[float] = _float_env("SENTIMENT_MICRO_HISTORY_RETENTION_DAYS", "7")

# --- Microstructure decision layer ---
# NOTE: these defaults are conservative starting points, NOT calibrated values.
# Record a few days of history, run `python -m sentiment_engine.storage.micro_stats`,
# then tune via env vars.
MICRO_QUALITY_MIN: Final[float] = _float_env("SENTIMENT_MICRO_QUALITY_MIN", "0.60")
MICRO_MIN_TRADES_60S: Final[int] = _int_env("SENTIMENT_MICRO_MIN_TRADES_60S", "30")
MICRO_CONFIRM_SCORE: Final[float] = _float_env("SENTIMENT_MICRO_CONFIRM_SCORE", "0.30")
MICRO_VETO_SCORE: Final[float] = _float_env("SENTIMENT_MICRO_VETO_SCORE", "0.45")
# Regime gates: a scalp into a wide spread or a volatility burst is vetoed/capped
# regardless of direction.
MICRO_MAX_SPREAD_BPS: Final[float] = _float_env("SENTIMENT_MICRO_MAX_SPREAD_BPS", "8.0")
MICRO_VOL_BURST_BPS: Final[float] = _float_env("SENTIMENT_MICRO_VOL_BURST_BPS", "30.0")
# Min near-price depth (quote value) during a volatility burst. 0 disables the
# depth check (depth scales vary per symbol; calibrate before enabling).
MICRO_MIN_DEPTH_QUOTE: Final[float] = _float_env("SENTIMENT_MICRO_MIN_DEPTH_QUOTE", "0")
# Sentiment adjustment applied on top of the micro decision (secondary layer).
SENTIMENT_MODIFIER_STEP: Final[Decimal] = _decimal_env("SENTIMENT_MODIFIER_STEP", "0.05")

# --- Outcome tracking ---
# Horizons (seconds) at which post-evaluation prices are recorded.
OUTCOME_HORIZONS_SECONDS: Final[Tuple[int, ...]] = tuple(
    int(v) for v in _tuple_env("SENTIMENT_OUTCOME_HORIZONS", "30,120,300")
)

# --- WebSocket session management ---
# Binance force-closes connections every 24h; reconnect gracefully before that.
MAX_WS_SESSION_SECONDS: Final[float] = _float_env("SENTIMENT_MAX_WS_SESSION_SECONDS", "43200")

# --- News RSS connector ---
NEWS_RSS_URLS: Final[Tuple[str, ...]] = _tuple_env(
    "SENTIMENT_NEWS_RSS_URLS",
    # General multi-coin crypto feeds (symbol inferred from each headline).
    "https://www.coindesk.com/arc/outboundfeeds/rss/,"
    "https://cointelegraph.com/rss,"
    "https://decrypt.co/feed,"
    "https://www.theblock.co/rss.xml,"
    "https://bitcoinmagazine.com/feed,"
    "https://cryptoslate.com/feed/,"
    "https://www.newsbtc.com/feed/,"
    "https://u.today/rss,"
    "https://cryptopanic.com/news/rss/,"
    # Coin-targeted feeds (every item implies one coin - see NEWS_IMPLIED_SYMBOLS).
    "https://cointelegraph.com/rss/tag/cardano,"
    "https://cointelegraph.com/rss/tag/bitcoin,"
    "https://cointelegraph.com/rss/tag/ethereum,"
    "https://news.google.com/rss/search?q=Cardano+ADA+crypto&hl=en-US&gl=US&ceid=US:en,"
    "https://news.google.com/rss/search?q=Ethereum+ETH+crypto&hl=en-US&gl=US&ceid=US:en,"
    "https://news.google.com/rss/search?q=BNB+Binance+Coin+crypto&hl=en-US&gl=US&ceid=US:en,"
    "https://news.google.com/rss/search?q=Solana+SOL+crypto&hl=en-US&gl=US&ceid=US:en,"
    "https://news.google.com/rss/search?q=XRP+Ripple+crypto&hl=en-US&gl=US&ceid=US:en,"
    "https://news.google.com/rss/search?q=Dogecoin+DOGE+crypto&hl=en-US&gl=US&ceid=US:en,"
    "https://news.google.com/rss/search?q=Chainlink+LINK+crypto&hl=en-US&gl=US&ceid=US:en",
)
NEWS_POLL_SECONDS: Final[float] = _float_env("SENTIMENT_NEWS_POLL_SECONDS", "60")

# Feeds whose every item is about a single coin, so items are tagged to that
# coin even when the headline does not literally name it (mirrors Reddit's
# IMPLIED_SYMBOL). Keys must match entries in NEWS_RSS_URLS exactly.
NEWS_IMPLIED_SYMBOLS: Final[dict] = {
    "https://bitcoinmagazine.com/feed": "BTC/USDT",
    "https://cointelegraph.com/rss/tag/cardano": "ADA/USDT",
    "https://cointelegraph.com/rss/tag/bitcoin": "BTC/USDT",
    "https://cointelegraph.com/rss/tag/ethereum": "ETH/USDT",
    "https://news.google.com/rss/search?q=Cardano+ADA+crypto&hl=en-US&gl=US&ceid=US:en": "ADA/USDT",
    "https://news.google.com/rss/search?q=Ethereum+ETH+crypto&hl=en-US&gl=US&ceid=US:en": "ETH/USDT",
    "https://news.google.com/rss/search?q=BNB+Binance+Coin+crypto&hl=en-US&gl=US&ceid=US:en": "BNB/USDT",
    "https://news.google.com/rss/search?q=Solana+SOL+crypto&hl=en-US&gl=US&ceid=US:en": "SOL/USDT",
    "https://news.google.com/rss/search?q=XRP+Ripple+crypto&hl=en-US&gl=US&ceid=US:en": "XRP/USDT",
    "https://news.google.com/rss/search?q=Dogecoin+DOGE+crypto&hl=en-US&gl=US&ceid=US:en": "DOGE/USDT",
    "https://news.google.com/rss/search?q=Chainlink+LINK+crypto&hl=en-US&gl=US&ceid=US:en": "LINK/USDT",
}

# --- Medium-term sentiment (persistent, multi-horizon) ---
SOCIAL_HISTORY_RETENTION_DAYS: Final[float] = _float_env("SENTIMENT_SOCIAL_HISTORY_RETENTION_DAYS", "14")
# Outlook horizons in seconds and their weights in the composite score.
OUTLOOK_HORIZONS_SECONDS_T: Final[Tuple[int, ...]] = tuple(
    int(v) for v in _tuple_env("SENTIMENT_OUTLOOK_HORIZONS", "3600,21600,86400")
)
OUTLOOK_HORIZON_WEIGHTS: Final[Tuple[float, ...]] = tuple(
    float(v) for v in _tuple_env("SENTIMENT_OUTLOOK_WEIGHTS", "0.5,0.3,0.2")
)
# News carries more signal per item than a single social post.
NEWS_SOURCE_WEIGHT: Final[float] = _float_env("SENTIMENT_NEWS_SOURCE_WEIGHT", "2.0")

# --- CryptoPanic (free news aggregator + crowd bullish/bearish votes) ---
# Dormant unless a free auth token (cryptopanic.com/developers) is provided.
CRYPTOPANIC_TOKEN: Final[str] = _str_env("SENTIMENT_CRYPTOPANIC_TOKEN", "")
CRYPTOPANIC_URL: Final[str] = _str_env("SENTIMENT_CRYPTOPANIC_URL", "https://cryptopanic.com/api/v1/posts/")
CRYPTOPANIC_POLL_SECONDS: Final[float] = _float_env("SENTIMENT_CRYPTOPANIC_POLL_SECONDS", "120")

# --- Optional transformer sentiment scorer (off by default) ---
# Set to a Hugging Face model id (e.g. "ElKulako/cryptobert" or "ProsusAI/finbert")
# to score text with a transformer instead of the lexicon. Needs `transformers`
# + `torch` installed; falls back to the lexicon automatically if unavailable.
TRANSFORMER_MODEL: Final[str] = _str_env("SENTIMENT_TRANSFORMER_MODEL", "")

# --- Optional ONNX-int8 scorer (lightweight, no PyTorch) ---
# Point SENTIMENT_ONNX_MODEL at a quantized .onnx (see convert_cryptobert_onnx.py)
# and SENTIMENT_ONNX_TOKENIZER at its tokenizer.json. Needs only `onnxruntime` +
# `tokenizers` at runtime (~350 MB total vs ~1.5-2 GB for PyTorch). Falls back to
# the lexicon if unavailable. Takes priority over the PyTorch transformer.
ONNX_MODEL_PATH: Final[str] = _str_env("SENTIMENT_ONNX_MODEL", "")
ONNX_TOKENIZER_PATH: Final[str] = _str_env("SENTIMENT_ONNX_TOKENIZER", "")
ONNX_LABELS: Final[str] = _str_env("SENTIMENT_ONNX_LABELS", "bearish,neutral,bullish")

# --- Market context (keyless) ---
FNG_URL: Final[str] = os.getenv("SENTIMENT_FNG_URL", "https://api.alternative.me/fng/?limit=1")
COINGECKO_GLOBAL_URL: Final[str] = os.getenv(
    "SENTIMENT_COINGECKO_GLOBAL_URL", "https://api.coingecko.com/api/v3/global"
)
CONTEXT_POLL_SECONDS: Final[float] = _float_env("SENTIMENT_CONTEXT_POLL_SECONDS", "900")
CONTEXT_STALE_SECONDS: Final[float] = _float_env("SENTIMENT_CONTEXT_STALE_SECONDS", "3600")
# Context may tilt an outlook by at most this much (it never dominates).
CONTEXT_MAX_TILT: Final[float] = _float_env("SENTIMENT_CONTEXT_MAX_TILT", "0.10")

# --- Reddit ---
# Keyless public JSON works only from residential IPs at low rates and is now
# frequently 403/429'd. Setting an app id+secret (free "script" app at
# https://www.reddit.com/prefs/apps) switches to userless OAuth (oauth.reddit.com,
# ~100 req/min) which is the reliable path. Empty -> keyless fallback.
REDDIT_SUBREDDITS: Final[Tuple[str, ...]] = _tuple_env(
    "SENTIMENT_REDDIT_SUBREDDITS", "CryptoCurrency,Bitcoin,cardano"
)
REDDIT_POLL_SECONDS: Final[float] = _float_env("SENTIMENT_REDDIT_POLL_SECONDS", "120")
REDDIT_CLIENT_ID: Final[str] = _str_env("SENTIMENT_REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: Final[str] = _str_env("SENTIMENT_REDDIT_CLIENT_SECRET", "")

# --- Bluesky (keyless Jetstream firehose, free, no auth) ---
BLUESKY_JETSTREAM_URL: Final[str] = _str_env(
    "SENTIMENT_BLUESKY_JETSTREAM_URL",
    "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post",
)

# --- Shock detection ---
# Recent activity window vs the baseline hour before it.
SHOCK_WINDOW_SECONDS: Final[float] = _float_env("SENTIMENT_SHOCK_WINDOW_SECONDS", "600")
SHOCK_BASELINE_SECONDS: Final[float] = _float_env("SENTIMENT_SHOCK_BASELINE_SECONDS", "3600")
# Message rate must be at least this multiple of baseline rate.
SHOCK_RATE_RATIO: Final[float] = _float_env("SENTIMENT_SHOCK_RATE_RATIO", "3.0")
# And mean sentiment in the window must be at least this extreme.
SHOCK_SENTIMENT_ABS: Final[float] = _float_env("SENTIMENT_SHOCK_SENTIMENT_ABS", "0.25")
# Minimum events in the window before a shock can fire at all.
SHOCK_MIN_EVENTS: Final[int] = _int_env("SENTIMENT_SHOCK_MIN_EVENTS", "8")

# Sources excluded from medium-term analytics (outlook, shock detection).
# Dashboard test injections stay visible in short-term snapshots but must not
# pollute the real sentiment record.
SYNTHETIC_SOURCES: Final[Tuple[str, ...]] = _tuple_env(
    "SENTIMENT_SYNTHETIC_SOURCES", "dashboard,dashboard-scenario,dashboard-baseline,manual"
)

# --- X/Twitter connector (key-gated, pay-per-use API) ---
# Dormant unless a bearer token is provided. Reads cost money ($0.005/tweet
# on pay-per-use), so a hard daily read budget is enforced client-side.
X_BEARER_TOKEN: Final[str] = os.getenv("SENTIMENT_X_BEARER_TOKEN", os.getenv("X_BEARER_TOKEN", ""))
X_SEARCH_URL: Final[str] = os.getenv("SENTIMENT_X_SEARCH_URL", "https://api.x.com/2/tweets/search/recent")
X_POLL_SECONDS: Final[float] = _float_env("SENTIMENT_X_POLL_SECONDS", "900")
X_MAX_RESULTS: Final[int] = _int_env("SENTIMENT_X_MAX_RESULTS", "10")
X_DAILY_READ_BUDGET: Final[int] = _int_env("SENTIMENT_X_DAILY_READS", "400")

"""Internal types for the standalone sentiment engine."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional


class TradeDirection(str, Enum):
    LONG = "STRAT_LONG"
    SHORT = "STRAT_SHORT"


class SentimentAction(str, Enum):
    CONFIRM = "confirm"
    NEUTRAL = "neutral"
    VETO = "veto"


@dataclass(frozen=True, slots=True)
class SocialEvent:
    symbol: str
    source: str
    text: str
    timestamp: float
    author: Optional[str] = None
    sentiment: Optional[Decimal] = None
    confidence: Optional[Decimal] = None


@dataclass(frozen=True, slots=True)
class PriceEvent:
    symbol: str
    price: Decimal
    timestamp: float


@dataclass(frozen=True, slots=True)
class SentimentSnapshot:
    symbol: str
    window_seconds: int
    message_count: int
    previous_message_count: int
    average_sentiment: Decimal
    previous_average_sentiment: Decimal
    sentiment_velocity: Decimal
    attention_spike: Decimal
    price_change_pct: Decimal
    data_quality: Decimal


@dataclass(frozen=True, slots=True)
class TradeEvaluation:
    symbol: str
    action: SentimentAction
    sentiment_score: Decimal
    confidence_modifier: Decimal
    final_confidence_estimate: Decimal
    data_quality: Decimal
    reason: str
    safe_to_use: bool
    snapshot: SentimentSnapshot


@dataclass(frozen=True, slots=True)
class MicrostructureSnapshot:
    """Latest microstructure metrics for one symbol, pushed by a live feed.

    All metric fields are Optional: None means "unavailable", never zero.
    Floats are used (not Decimal) because these are statistical signals.
    """

    symbol: str
    computed_at: float  # unix seconds, set by the feed process
    last_price: Optional[float] = None
    buy_volume_10s: Optional[float] = None
    sell_volume_10s: Optional[float] = None
    trade_imbalance_10s: Optional[float] = None
    trade_imbalance_30s: Optional[float] = None
    trade_imbalance_60s: Optional[float] = None
    relative_volume: Optional[float] = None
    momentum_pct_30s: Optional[float] = None
    volatility_bps_30s: Optional[float] = None
    vwap_distance_bps: Optional[float] = None
    bid_depth_quote: Optional[float] = None
    ask_depth_quote: Optional[float] = None
    book_imbalance: Optional[float] = None
    spread_bps: Optional[float] = None
    trade_count_60s: Optional[int] = None


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """Global market context from keyless public sources."""

    fetched_at: float
    fear_greed_value: Optional[int] = None      # 0..100
    fear_greed_label: Optional[str] = None      # "Extreme Fear" .. "Extreme Greed"
    btc_dominance_pct: Optional[float] = None
    total_market_cap_usd: Optional[float] = None
    market_cap_change_24h_pct: Optional[float] = None

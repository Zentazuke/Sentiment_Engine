"""Symbol normalization and lightweight coin mention helpers."""

from __future__ import annotations

from typing import Iterable

from sentiment_engine.config import SUPPORTED_SYMBOLS

_SYMBOL_ALIASES = {
    "BTC/USDT": ("btc", "bitcoin", "xbt"),
    "ADA/USDT": ("ada", "cardano"),
    "ETH/USDT": ("eth", "ethereum"),
    # Deliberately NOT bare "binance" (matches the exchange, not the coin).
    "BNB/USDT": ("bnb", "binance coin", "binancecoin"),
    "SOL/USDT": ("sol", "solana"),
    "XRP/USDT": ("xrp", "ripple"),
    "DOGE/USDT": ("doge", "dogecoin"),
    # Deliberately NOT bare "link" (matches hyperlinks, "link" everywhere).
    "LINK/USDT": ("chainlink",),
}


def normalize_symbol(symbol_or_key: str) -> str:
    value = symbol_or_key.strip().upper().replace("-", "/")
    if "/" not in value and value.endswith("USDT"):
        value = f"{value[:-4]}/USDT"
    if value not in SUPPORTED_SYMBOLS:
        raise ValueError(f"Unsupported symbol {symbol_or_key!r}. Supported: {SUPPORTED_SYMBOLS}")
    return value


def symbol_key(symbol: str) -> str:
    return normalize_symbol(symbol).replace("/", "-")


def aliases_for(symbol: str) -> Iterable[str]:
    return _SYMBOL_ALIASES.get(normalize_symbol(symbol), (normalize_symbol(symbol).split("/")[0].lower(),))

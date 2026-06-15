"""Fast rule-based sentiment scoring, v2.

Still deliberately cheap (no models, no network), but smarter than v1:
- weighted crypto lexicon (words and multi-word phrases)
- negation flips ("not bullish", "no breakout", "fake pump")
- intensifiers amplify nearby sentiment ("dumping hard", "massive pump")
- hedging dampens score and confidence ("might", "maybe", "not sure")
- emoji signals, question-mark uncertainty, mixed-signal confidence penalty

Public contract (unchanged from v1):
    score_text(text) -> (sentiment Decimal in [-1, 1], confidence Decimal in [0, 1])
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Dict, Final, List, Tuple

from sentiment_engine.processing.text_cleaner import clean_text

# Multi-word phrases are matched first and masked out so their component
# words cannot double-count. Weights are bearish(-1)..bullish(+1).
PHRASES: Final[Dict[str, float]] = {
    "short squeeze": 0.7,
    "support held": 0.7,
    "buyers stepping in": 0.6,
    "all time high": 0.6,
    "buy the dip": 0.4,
    "bear trap": 0.4,
    "support lost": -0.8,
    "rug pull": -0.9,
    "bull trap": -0.7,
    "dead cat": -0.6,
    "fake breakout": -0.6,
    "panic sell": -0.8,
    "sell off": -0.7,
    "no direction": -0.1,
}

WORDS: Final[Dict[str, float]] = {
    # bullish
    "breakout": 0.6, "pump": 0.5, "pumping": 0.6, "bullish": 0.7, "bulls": 0.4,
    "moon": 0.6, "mooning": 0.7, "strong": 0.4, "squeeze": 0.4, "accumulation": 0.4,
    "ath": 0.5, "ripping": 0.6, "bounce": 0.3, "reclaim": 0.5, "rally": 0.5,
    "rallies": 0.5, "rallying": 0.6, "surge": 0.6, "surges": 0.6, "surging": 0.6,
    "soar": 0.6, "soars": 0.6, "soaring": 0.6, "green": 0.3, "higher": 0.3,
    "undervalued": 0.4, "oversold": 0.3, "recovery": 0.4, "rebound": 0.4,
    "upgrade": 0.3, "adoption": 0.4, "inflow": 0.4, "inflows": 0.4, "hodl": 0.2,
    "gains": 0.4, "winning": 0.4, "breakouts": 0.6, "parabolic": 0.5, "vertical": 0.4,
    # bearish
    "dump": -0.6, "dumping": -0.7, "dumps": -0.6, "crash": -0.8, "crashes": -0.8,
    "crashing": -0.8, "bearish": -0.7, "bears": -0.4, "rekt": -0.7, "selloff": -0.7,
    "rejection": -0.5, "rejected": -0.5, "panic": -0.8, "rug": -0.8, "rugpull": -0.9,
    "liquidated": -0.6, "liquidation": -0.5, "liquidations": -0.5, "capitulation": -0.7,
    "red": -0.3, "weak": -0.4, "downside": -0.5, "lower": -0.3, "fud": -0.4,
    "scam": -0.7, "hack": -0.8, "hacked": -0.9, "exploit": -0.7, "exploited": -0.8,
    "lawsuit": -0.5, "ban": -0.6, "banned": -0.6, "overvalued": -0.4, "overbought": -0.3,
    "fear": -0.4, "plunge": -0.7, "plunges": -0.7, "plunging": -0.7, "tank": -0.6,
    "tanking": -0.7, "bleed": -0.5, "bleeding": -0.6, "correction": -0.3,
    "breakdown": -0.5, "outflow": -0.4, "outflows": -0.4, "crisis": -0.6,
    "collapse": -0.8, "collapses": -0.8, "ugly": -0.4, "losses": -0.4, "losing": -0.4,
    "choppy": -0.2, "sideways": -0.1, "insolvent": -0.8, "bankruptcy": -0.8,
}

EMOJIS: Final[Dict[str, float]] = {
    "🚀": 0.6, "📈": 0.5, "🟢": 0.3, "💎": 0.3, "🐂": 0.5, "🔥": 0.3,
    "📉": -0.5, "🔻": -0.4, "🩸": -0.5, "💀": -0.6, "🐻": -0.5, "⚠️": -0.3,
}

# Negators flip the following sentiment word (looking back up to 3 tokens).
NEGATORS: Final[Tuple[str, ...]] = (
    "not", "no", "never", "hardly", "barely", "isnt", "isn't", "wasnt", "wasn't",
    "wont", "won't", "doesnt", "doesn't", "without", "fake", "false", "fading",
)
# Intensifiers amplify sentiment words within 2 tokens (either side).
INTENSIFIERS: Final[Tuple[str, ...]] = (
    "insane", "insanely", "massive", "massively", "huge", "crazy", "violent",
    "violently", "hard", "explosive", "super", "extremely", "very", "mega",
    "absolutely", "brutal", "brutally",
)
# Hedges signal uncertainty: dampen the score and the confidence.
HEDGES: Final[Tuple[str, ...]] = (
    "might", "maybe", "possibly", "perhaps", "could", "unsure", "undecided",
    "potentially", "rumor", "rumour", "allegedly",
)

_TOKEN_RE = re.compile(r"[a-z']+")
_NEGATION_LOOKBACK = 3
_INTENSIFIER_RANGE = 2
_INTENSIFIER_FACTOR = 1.3
_NEGATION_FACTOR = -0.8
_HEDGE_FACTOR = 0.5


def _phrase_contributions(cleaned: str) -> Tuple[List[float], str]:
    """Score multi-word phrases and mask them out of the text."""
    contributions: List[float] = []
    masked = cleaned
    for phrase, weight in PHRASES.items():
        count = masked.count(phrase)
        if count:
            contributions.extend([weight] * count)
            masked = masked.replace(phrase, " ")
    return contributions, masked


def _token_contributions(tokens: List[str]) -> List[float]:
    contributions: List[float] = []
    for index, token in enumerate(tokens):
        weight = WORDS.get(token)
        if weight is None:
            continue
        lookback = tokens[max(0, index - _NEGATION_LOOKBACK):index]
        if any(negator in lookback for negator in NEGATORS):
            weight *= _NEGATION_FACTOR
        nearby = tokens[max(0, index - _INTENSIFIER_RANGE):index + _INTENSIFIER_RANGE + 1]
        if any(term in nearby for term in INTENSIFIERS):
            weight *= _INTENSIFIER_FACTOR
        contributions.append(weight)
    return contributions


def score_text(text: str) -> tuple[Decimal, Decimal]:
    """Return (sentiment, confidence). Sentiment -1..1, confidence 0..1."""
    cleaned = clean_text(text)
    if not cleaned:
        return Decimal("0"), Decimal("0")

    phrase_scores, masked = _phrase_contributions(cleaned)
    tokens = _TOKEN_RE.findall(masked)
    contributions = phrase_scores + _token_contributions(tokens)
    contributions += [weight for emoji, weight in EMOJIS.items() if emoji in text]

    if not contributions:
        return Decimal("0"), Decimal("0.20")

    hedged = any(term in tokens for term in HEDGES) or "not sure" in cleaned
    total = sum(contributions)
    if hedged:
        total *= _HEDGE_FACTOR
    score = max(-1.0, min(1.0, total))

    # Confidence: more evidence -> higher; mixed signals, hedging and
    # questions all reduce it.
    abs_sum = sum(abs(weight) for weight in contributions)
    purity = abs(total) / abs_sum if abs_sum > 0 else 0.0
    confidence = min(0.90, 0.35 + 0.12 * len(contributions))
    confidence *= 0.6 + 0.4 * purity
    if hedged:
        confidence -= 0.15
    if "?" in text:
        confidence *= 0.8
    confidence = max(0.05, min(0.90, confidence))

    return (
        Decimal(str(round(score, 4))),
        Decimal(str(round(confidence, 4))),
    )

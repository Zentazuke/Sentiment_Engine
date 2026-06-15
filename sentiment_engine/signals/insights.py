"""Descriptive insights over the social/news record.

Pure functions answering "WHY is the mood what it is":
- trending_terms: which words dominate recent chatter, with their mood
- source_breakdown: news vs reddit vs other - do press and retail agree?

Analytics only; nothing here influences decisions.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

# (timestamp, source, text, sentiment)
TextRow = Tuple[float, str, str, Optional[float]]

_TOKEN_RE = re.compile(r"[a-z][a-z'\-]{2,}")

# Common words + coin names (redundant - every row is already about the coin).
STOPWORDS = frozenset("""
the and for with that this from are was were has have had you your our their its
not but they them then than what when where which who why how all any can could
will would should just like get got going some more most much very out about into
over under after before between during off only own same say said new now today
btc bitcoin ada cardano crypto cryptocurrency coin coins token tokens price market
been being doing does did because while still also even back down up his her him
one two three first next last week month year day days via amid among though
""".split())


def trending_terms(rows: Sequence[TextRow], top_n: int = 15) -> List[Dict[str, object]]:
    """Most frequent meaningful words, each with the avg sentiment of the
    messages containing it."""
    counts: Dict[str, int] = defaultdict(int)
    sentiments: Dict[str, List[float]] = defaultdict(list)
    for _ts, _source, text, sentiment in rows:
        seen_in_message = set()
        for token in _TOKEN_RE.findall(text.lower()):
            if token in STOPWORDS or token in seen_in_message:
                continue
            seen_in_message.add(token)
            counts[token] += 1
            if sentiment is not None:
                sentiments[token].append(sentiment)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    return [
        {
            "term": term,
            "count": count,
            "avg_sentiment": round(sum(sentiments[term]) / len(sentiments[term]), 4)
            if sentiments[term] else None,
        }
        for term, count in ranked
        if count >= 2  # a word seen once is noise, not a trend
    ]


def _source_class(source: str) -> str:
    if source.startswith("news:"):
        return "news"
    if source.startswith("reddit:"):
        return "reddit"
    return "other"


def source_breakdown(rows: Sequence[TextRow]) -> Dict[str, Dict[str, object]]:
    """Count + average sentiment per source class, plus their divergence."""
    grouped: Dict[str, List[float]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)
    for _ts, source, _text, sentiment in rows:
        cls = _source_class(source)
        counts[cls] += 1
        if sentiment is not None:
            grouped[cls].append(sentiment)
    result: Dict[str, Dict[str, object]] = {}
    for cls in ("news", "reddit", "other"):
        values = grouped.get(cls, [])
        result[cls] = {
            "count": counts.get(cls, 0),
            "avg_sentiment": round(sum(values) / len(values), 4) if values else None,
        }
    news_avg = result["news"]["avg_sentiment"]
    reddit_avg = result["reddit"]["avg_sentiment"]
    result["divergence"] = (
        round(abs(news_avg - reddit_avg), 4)
        if news_avg is not None and reddit_avg is not None else None
    )
    return result

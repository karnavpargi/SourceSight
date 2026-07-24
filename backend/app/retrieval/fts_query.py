"""Normalize agent search strings for Postgres full-text search."""

from __future__ import annotations

import re

_TICKER_PATTERN = re.compile(r"\b(AAPL|AMZN|GOOGL|MSFT|NVDA)\b", re.IGNORECASE)
_YEAR_PATTERN = re.compile(r"\b20\d{2}\b")
_SECTION_PATTERN = re.compile(r"\bitem\s+1a\b", re.IGNORECASE)
_FORM_PATTERN = re.compile(r"\b10-?k\b", re.IGNORECASE)
_MAX_FTS_TERMS = 8


def parse_full_text_query(query: str) -> tuple[str, list[str]]:
    """Strip metadata-like tokens and cap length so websearch_to_tsquery can match."""
    tickers = sorted({match.upper() for match in _TICKER_PATTERN.findall(query)})

    text = query
    text = _YEAR_PATTERN.sub(" ", text)
    text = _SECTION_PATTERN.sub(" ", text)
    text = _FORM_PATTERN.sub(" ", text)
    text = _TICKER_PATTERN.sub(" ", text)

    seen: set[str] = set()
    words: list[str] = []
    for word in text.split():
        lowered = word.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        words.append(word)
        if len(words) >= _MAX_FTS_TERMS:
            break

    return " ".join(words).strip(), tickers


def fts_query_variants(query: str) -> list[tuple[str, list[str]]]:
    """Primary normalized query plus shorter fallbacks when the first returns no hits."""
    normalized, tickers = parse_full_text_query(query)
    if not normalized:
        return []

    variants: list[tuple[str, list[str]]] = [(normalized, tickers)]
    words = normalized.split()
    if len(words) > 4:
        variants.append((" ".join(words[:4]), tickers))
    if len(words) > 2:
        variants.append((" ".join(words[:2]), tickers))
    return variants

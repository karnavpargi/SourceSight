from __future__ import annotations

import pytest

from app.retrieval.fts_query import fts_query_variants, parse_full_text_query


def test_parse_full_text_query_strips_years_tickers_and_sections() -> None:
    query = "risk factors artificial intelligence Item 1A AAPL 2021 2022 2023 2024 2025"
    normalized, tickers = parse_full_text_query(query)
    assert tickers == ["AAPL"]
    assert "2021" not in normalized
    assert "1a" not in normalized.lower()
    assert "aapl" not in normalized.lower()
    assert "artificial" in normalized


def test_fts_query_variants_include_shorter_fallbacks() -> None:
    variants = fts_query_variants(
        "risk factors artificial intelligence export controls supply chain regulation cloud infrastructure"
    )
    assert variants[0][0].count(" ") <= 7
    assert len(variants) >= 2

"""Live-database retrieval integration tests."""

from __future__ import annotations

import time

import pytest

from app.retrieval.retriever import retrieve_passages
from ingest.db import session_scope

pytestmark = pytest.mark.integration


def _amazon_passages(passages):
    return [passage for passage in passages if passage.ticker == "AMZN"]


def _apple_passages(passages):
    return [passage for passage in passages if passage.ticker == "AAPL"]


def _mentions_aws_segment_results(content: str) -> bool:
    lower = content.lower()
    return "aws" in lower and ("operating" in lower or "segment" in lower)


def _mentions_iphone_or_product_revenue_mix(content: str) -> bool:
    lower = content.lower()
    product_terms = ("iphone", "services", "mac", "ipad", "wearable")
    return any(term in lower for term in product_terms) and "revenue" in lower


def test_aws_operating_income_returns_amazon_aws_segment_passages(
    ingested_corpus: None,
    ollama_embeddings: None,
) -> None:
    with session_scope() as session:
        result = retrieve_passages(session, "AWS operating income", limit=5)

    assert len(result.passages) == 5
    amazon_passages = _amazon_passages(result.passages)
    assert len(amazon_passages) == 5
    assert all(_mentions_aws_segment_results(passage.content) for passage in amazon_passages)
    assert len({passage.fiscal_year for passage in amazon_passages}) >= 3


def test_iphone_revenue_mix_shift_returns_apple_passages_across_years(
    ingested_corpus: None,
    ollama_embeddings: None,
) -> None:
    with session_scope() as session:
        result = retrieve_passages(session, "iPhone revenue mix shift", limit=10)

    apple_passages = _apple_passages(result.passages)
    assert len(apple_passages) >= 3
    assert len({passage.fiscal_year for passage in apple_passages}) >= 2
    assert any(_mentions_iphone_or_product_revenue_mix(passage.content) for passage in apple_passages)
    assert any(passage.ticker == "AAPL" for passage in result.passages[:5])


def test_hybrid_retrieval_p95_under_500ms(
    ingested_corpus: None,
    ollama_embeddings: None,
) -> None:
    timings_ms: list[float] = []
    with session_scope() as session:
        for _ in range(20):
            started = time.perf_counter()
            retrieve_passages(session, "AWS operating income", limit=10)
            timings_ms.append((time.perf_counter() - started) * 1000)

    timings_ms.sort()
    p95_ms = timings_ms[int(0.95 * len(timings_ms)) - 1]
    assert p95_ms < 500, f"p95 hybrid retrieval was {p95_ms:.1f}ms (limit 500ms)"

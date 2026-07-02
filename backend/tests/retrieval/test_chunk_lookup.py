from __future__ import annotations

from uuid import UUID

from pydantic_ai.exceptions import ModelRetry

from app.retrieval.chunk_lookup import chunk_not_found_retry

CHUNK_ID = UUID("11111111-1111-1111-1111-111111111111")


def test_chunk_not_found_retry_is_model_retry_with_guidance() -> None:
    err = chunk_not_found_retry(CHUNK_ID)

    assert isinstance(err, ModelRetry)
    assert str(CHUNK_ID) in str(err)
    assert "search_filings" in str(err)

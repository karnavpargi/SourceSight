from __future__ import annotations

from datetime import date
from unittest.mock import Mock
from uuid import UUID

import pytest
from pydantic_ai.exceptions import ModelRetry

from app.chat.orchestrator import _RecordingRetriever
from app.retrieval.types import SourcePassage

CHUNK_ID = UUID("11111111-1111-1111-1111-111111111111")
DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")


def _passage() -> SourcePassage:
    return SourcePassage(
        chunk_id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        chunk_index=3,
        content="Data Center demand increased.",
        section="Item 1. Business",
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001045810-24-000029",
        filing_date=date(2024, 2, 21),
        source_url="https://example.com/nvda-10k",
        score=0.42,
    )


def test_read_chunk_returns_cached_passage_without_db_lookup() -> None:
    passage = _passage()
    inner = Mock()
    recorder = _RecordingRetriever(inner=inner)
    recorder.seen[CHUNK_ID] = passage

    assert recorder.read_chunk(CHUNK_ID) == passage
    inner.read_chunk.assert_not_called()


def test_read_chunk_raises_model_retry_for_unknown_chunk_id() -> None:
    inner = Mock()
    inner.read_chunk.side_effect = ValueError(f"Chunk {CHUNK_ID} not found.")
    recorder = _RecordingRetriever(inner=inner)

    with pytest.raises(ModelRetry, match="search_filings"):
        recorder.read_chunk(CHUNK_ID)


def test_read_surrounding_chunks_falls_back_to_cached_anchor() -> None:
    passage = _passage()
    neighbor = _passage()
    inner = Mock()
    inner.read_surrounding_chunks.side_effect = ValueError(f"Chunk {CHUNK_ID} not found.")
    recorder = _RecordingRetriever(inner=inner)
    recorder.seen[CHUNK_ID] = passage
    recorder._neighbors_for_cached_anchor = Mock(return_value=[neighbor])  # type: ignore[method-assign]

    result = recorder.read_surrounding_chunks(CHUNK_ID, window=1)

    assert result == [neighbor]
    inner.read_surrounding_chunks.assert_called_once_with(CHUNK_ID, window=1)

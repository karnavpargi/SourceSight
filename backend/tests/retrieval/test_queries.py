from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from app.config import settings
from app.retrieval.queries import (
    RankedChunkHit,
    build_embedding_search_statement,
    build_full_text_search_statement,
    search_chunks_by_embedding,
    search_chunks_by_full_text,
)


def _mock_rows(*rows: tuple[uuid.UUID, float]) -> MagicMock:
    result = MagicMock()
    result.all.return_value = [
        type("Row", (), {"id": chunk_id, "distance": value, "score": value})()
        for chunk_id, value in rows
    ]
    return result


def test_search_chunks_by_embedding_maps_distance_to_similarity() -> None:
    chunk_id = uuid.uuid4()
    session = MagicMock()
    session.execute.return_value = _mock_rows((chunk_id, 0.25))

    hits = search_chunks_by_embedding(session, [0.1] * settings.embedding_dimensions, limit=5)

    assert hits == [RankedChunkHit(chunk_id=chunk_id, score=0.75)]


def test_search_chunks_by_embedding_rejects_wrong_dimensions() -> None:
    session = MagicMock()
    with pytest.raises(ValueError, match="expected"):
        search_chunks_by_embedding(session, [0.1, 0.2], limit=5)


def test_search_chunks_by_embedding_returns_empty_for_non_positive_limit() -> None:
    session = MagicMock()
    assert search_chunks_by_embedding(session, [0.0] * settings.embedding_dimensions, limit=0) == []
    session.execute.assert_not_called()


def test_search_chunks_by_full_text_maps_rank_scores() -> None:
    chunk_id = uuid.uuid4()
    session = MagicMock()
    session.execute.return_value = _mock_rows((chunk_id, 0.42))

    hits = search_chunks_by_full_text(session, "supply chain concentration", limit=10)

    assert hits == [RankedChunkHit(chunk_id=chunk_id, score=0.42)]


def test_search_chunks_by_full_text_returns_empty_for_blank_query() -> None:
    session = MagicMock()
    assert search_chunks_by_full_text(session, "   ", limit=10) == []
    session.execute.assert_not_called()


def test_search_chunks_by_full_text_returns_empty_for_non_positive_limit() -> None:
    session = MagicMock()
    assert search_chunks_by_full_text(session, "aws revenue", limit=0) == []
    session.execute.assert_not_called()


def test_build_embedding_search_statement_uses_cosine_distance() -> None:
    embedding = [0.0] * settings.embedding_dimensions
    statement = build_embedding_search_statement(embedding, limit=7)
    compiled = str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert "document_chunks" in compiled
    assert "embedding" in compiled
    assert "LIMIT 7" in compiled


def test_build_full_text_search_statement_uses_websearch_to_tsquery() -> None:
    statement = build_full_text_search_statement("iPhone revenue", limit=8)
    compiled = str(statement.compile())
    assert "websearch_to_tsquery" in compiled
    assert "@@" in compiled
    assert "ts_rank_cd" in compiled
    assert "document_chunks" in compiled

from __future__ import annotations

import uuid
from datetime import date
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.database.document_chunk import DocumentChunk
from app.database.source_document import SourceDocument
from app.retrieval.fusion import FusedChunkHit
from app.retrieval.queries import RankedChunkHit
from app.retrieval.retriever import (
    _default_embed_query,
    _load_chunks,
    _load_neighbor_passages,
    _to_source_passage,
    retrieve_passages,
)
from app.retrieval.types import SourcePassage


@pytest.fixture(autouse=True)
def enable_embeddings_for_retriever_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.retrieval.retriever.settings.embedding_provider", "google")


def _make_chunk(
    *,
    chunk_id: uuid.UUID,
    document: SourceDocument,
    chunk_index: int,
    content: str,
    section: str | None = None,
) -> DocumentChunk:
    chunk = DocumentChunk(
        id=chunk_id,
        document_id=document.id,
        chunk_index=chunk_index,
        page=None,
        section=section,
        content=content,
        token_count=10,
        embedding=None,
        metadata_={},
    )
    chunk.document = document
    return chunk


def _make_document(*, ticker: str = "AAPL", fiscal_year: int = 2024) -> SourceDocument:
    return SourceDocument(
        id=uuid.uuid4(),
        ticker=ticker,
        cik="0000320193",
        company_name="Apple Inc.",
        form_type="10-K",
        fiscal_year=fiscal_year,
        accession_number="0000320193-24-000123",
        filing_date=date(2024, 11, 1),
        report_date=date(2024, 9, 28),
        primary_document="aapl.htm",
        source_url="https://example.com/aapl",
        markdown_content="# filing",
        metadata_={},
    )


def test_retrieve_passages_returns_empty_for_blank_query() -> None:
    session = MagicMock()
    result = retrieve_passages(session, "   ")
    assert result.query == "   "
    assert result.passages == []
    session.execute.assert_not_called()


def test_retrieve_passages_hybrid_search_and_fusion() -> None:
    session = MagicMock()
    document = _make_document()
    chunk_primary = _make_chunk(
        chunk_id=uuid.uuid4(),
        document=document,
        chunk_index=3,
        content="AWS operating income increased.",
        section="Item 7. MD&A",
    )
    chunk_neighbor = _make_chunk(
        chunk_id=uuid.uuid4(),
        document=document,
        chunk_index=4,
        content="Segment margin detail.",
        section="Item 7. MD&A",
    )

    vector_hits = [RankedChunkHit(chunk_id=chunk_primary.id, score=0.9)]
    text_hits = [RankedChunkHit(chunk_id=chunk_primary.id, score=0.4)]
    fused_hits = [FusedChunkHit(chunk_id=chunk_primary.id, score=0.03)]

    call_count = 0

    def fake_execute(statement):
        nonlocal call_count
        call_count += 1
        result = MagicMock()
        scalars = MagicMock()
        if call_count == 1:
            scalars.all.return_value = [chunk_primary]
        else:
            scalars.all.return_value = [chunk_neighbor]
        result.scalars.return_value = scalars
        return result

    session.execute.side_effect = fake_execute

    with patch(
        "app.retrieval.retriever.search_chunks_by_embedding",
        return_value=vector_hits,
    ) as mock_vector, patch(
        "app.retrieval.retriever.search_chunks_by_full_text",
        return_value=text_hits,
    ) as mock_text, patch(
        "app.retrieval.retriever.reciprocal_rank_fusion",
        return_value=fused_hits,
    ) as mock_fusion:
        result = retrieve_passages(
            session,
            "AWS operating income",
            embed_query=lambda _query: [0.1] * 768,
            neighbor_window=1,
        )

    mock_vector.assert_called_once()
    mock_text.assert_called_once()
    mock_fusion.assert_called_once()
    assert result.query == "AWS operating income"
    assert len(result.passages) == 2
    assert result.passages[0] == SourcePassage(
        chunk_id=chunk_primary.id,
        document_id=document.id,
        chunk_index=3,
        content="AWS operating income increased.",
        section="Item 7. MD&A",
        page=None,
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0000320193-24-000123",
        filing_date=date(2024, 11, 1),
        report_date=date(2024, 9, 28),
        source_url="https://example.com/aapl",
        score=0.03,
        is_neighbor=False,
    )
    assert result.passages[1].chunk_id == chunk_neighbor.id
    assert result.passages[1].is_neighbor is True
    assert result.passages[1].score == 0.0


def test_retrieve_passages_preserves_fused_ranking() -> None:
    session = MagicMock()
    document = _make_document(ticker="AMZN")
    chunk_a = _make_chunk(
        chunk_id=uuid.uuid4(),
        document=document,
        chunk_index=1,
        content="Passage A",
    )
    chunk_b = _make_chunk(
        chunk_id=uuid.uuid4(),
        document=document,
        chunk_index=2,
        content="Passage B",
    )

    fused_hits = [
        FusedChunkHit(chunk_id=chunk_b.id, score=0.05),
        FusedChunkHit(chunk_id=chunk_a.id, score=0.04),
    ]

    def fake_execute(_statement):
        result = MagicMock()
        scalars = MagicMock()
        scalars.all.return_value = [chunk_a, chunk_b]
        result.scalars.return_value = scalars
        return result

    session.execute.side_effect = fake_execute

    with patch("app.retrieval.retriever.search_chunks_by_embedding", return_value=[]), patch(
        "app.retrieval.retriever.search_chunks_by_full_text", return_value=[]
    ), patch("app.retrieval.retriever.reciprocal_rank_fusion", return_value=fused_hits):
        result = retrieve_passages(
            session,
            "Amazon AWS",
            embed_query=lambda _query: [0.0] * 768,
        )

    assert [passage.chunk_id for passage in result.passages] == [chunk_b.id, chunk_a.id]


def test_retrieve_passages_returns_empty_when_fusion_finds_nothing() -> None:
    session = MagicMock()

    with patch("app.retrieval.retriever.search_chunks_by_embedding", return_value=[]), patch(
        "app.retrieval.retriever.search_chunks_by_full_text", return_value=[]
    ), patch("app.retrieval.retriever.reciprocal_rank_fusion", return_value=[]):
        result = retrieve_passages(
            session,
            "Amazon AWS",
            embed_query=lambda _query: [0.0] * 768,
        )

    assert result.query == "Amazon AWS"
    assert result.passages == []
    session.execute.assert_not_called()


def test_retrieve_passages_skips_vector_search_when_embeddings_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    monkeypatch.setattr("app.retrieval.retriever.settings.embedding_provider", "none")

    with patch(
        "app.retrieval.retriever.search_chunks_by_embedding",
    ) as mock_vector, patch(
        "app.retrieval.retriever.search_chunks_by_full_text",
        return_value=[],
    ), patch(
        "app.retrieval.retriever.reciprocal_rank_fusion",
        return_value=[],
    ):
        result = retrieve_passages(session, "AWS operating income")

    mock_vector.assert_not_called()
    assert result.passages == []


def test_retrieve_passages_falls_back_to_full_text_when_embedding_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.retrieval.retriever.settings.embedding_provider", "google")
    session = MagicMock()
    document = _make_document()
    chunk = _make_chunk(
        chunk_id=uuid.uuid4(),
        document=document,
        chunk_index=1,
        content="AWS operating income increased.",
    )
    text_hits = [RankedChunkHit(chunk_id=chunk.id, score=0.4)]
    fused_hits = [FusedChunkHit(chunk_id=chunk.id, score=0.016)]

    def fake_execute(_statement):
        result = MagicMock()
        scalars = MagicMock()
        scalars.all.return_value = [chunk]
        result.scalars.return_value = scalars
        return result

    session.execute.side_effect = fake_execute

    with patch(
        "app.retrieval.retriever.search_chunks_by_embedding",
    ) as mock_vector, patch(
        "app.retrieval.retriever.search_chunks_by_full_text",
        return_value=text_hits,
    ) as mock_text, patch(
        "app.retrieval.retriever.reciprocal_rank_fusion",
        return_value=fused_hits,
    ) as mock_fusion:
        result = retrieve_passages(
            session,
            "AWS operating income",
            embed_query=lambda _query: (_ for _ in ()).throw(
                httpx.ConnectError("connection refused")
            ),
        )

    mock_vector.assert_not_called()
    mock_text.assert_called_once()
    mock_fusion.assert_called_once_with([], text_hits, limit=10)
    assert len(result.passages) == 1
    assert result.passages[0].chunk_id == chunk.id


def test_retrieve_passages_uses_default_embed_query() -> None:
    session = MagicMock()

    with patch("app.retrieval.retriever.search_chunks_by_embedding", return_value=[]), patch(
        "app.retrieval.retriever.search_chunks_by_full_text", return_value=[]
    ), patch("app.retrieval.retriever.reciprocal_rank_fusion", return_value=[]), patch(
        "app.retrieval.retriever.embed_query_text",
        return_value=[0.5] * 768,
    ) as embed_query:
        retrieve_passages(session, "Amazon AWS")

    embed_query.assert_called_once_with("Amazon AWS")


def test_default_embed_query_delegates_to_embed_query() -> None:
    with patch(
        "app.retrieval.retriever.embed_query_text",
        return_value=[0.1, 0.2],
    ) as embed_query:
        assert _default_embed_query("aws revenue") == [0.1, 0.2]
    embed_query.assert_called_once_with("aws revenue")


def test_load_chunks_returns_empty_for_no_ids() -> None:
    session = MagicMock()
    assert _load_chunks(session, []) == {}
    session.execute.assert_not_called()


def test_load_neighbor_passages_returns_empty_for_zero_window() -> None:
    session = MagicMock()
    document = _make_document()
    chunk = _make_chunk(
        chunk_id=uuid.uuid4(),
        document=document,
        chunk_index=3,
        content="Primary passage.",
    )
    passage = _to_source_passage(chunk, score=0.5)

    neighbors = _load_neighbor_passages(
        session,
        [passage],
        neighbor_window=0,
        existing_chunk_ids={chunk.id},
    )

    assert neighbors == []
    session.execute.assert_not_called()


def test_load_neighbor_passages_skips_existing_and_out_of_spec_chunks() -> None:
    session = MagicMock()
    document = _make_document()
    primary = _make_chunk(
        chunk_id=uuid.uuid4(),
        document=document,
        chunk_index=3,
        content="Primary passage.",
    )
    existing_neighbor = _make_chunk(
        chunk_id=uuid.uuid4(),
        document=document,
        chunk_index=2,
        content="Already loaded.",
    )
    matching_neighbor = _make_chunk(
        chunk_id=uuid.uuid4(),
        document=document,
        chunk_index=4,
        content="Neighbor passage.",
    )
    out_of_spec = _make_chunk(
        chunk_id=uuid.uuid4(),
        document=document,
        chunk_index=99,
        content="Unrelated chunk.",
    )

    result = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = [existing_neighbor, matching_neighbor, out_of_spec]
    result.scalars.return_value = scalars
    session.execute.return_value = result

    passage = _to_source_passage(primary, score=0.5)
    neighbors = _load_neighbor_passages(
        session,
        [passage],
        neighbor_window=1,
        existing_chunk_ids={primary.id, existing_neighbor.id},
    )

    assert len(neighbors) == 1
    assert neighbors[0].chunk_id == matching_neighbor.id
    assert neighbors[0].is_neighbor is True


def test_to_source_passage_raises_when_document_missing() -> None:
    chunk = _make_chunk(
        chunk_id=uuid.uuid4(),
        document=_make_document(),
        chunk_index=1,
        content="Orphan chunk.",
    )
    chunk.document = None

    with pytest.raises(ValueError, match="missing its source document"):
        _to_source_passage(chunk, score=0.0)


def test_retrieve_passages_skips_missing_chunk_ids_from_database() -> None:
    session = MagicMock()
    missing_id = uuid.uuid4()
    fused_hits = [FusedChunkHit(chunk_id=missing_id, score=0.03)]

    def fake_execute(_statement):
        result = MagicMock()
        scalars = MagicMock()
        scalars.all.return_value = []
        result.scalars.return_value = scalars
        return result

    session.execute.side_effect = fake_execute

    with patch("app.retrieval.retriever.search_chunks_by_embedding", return_value=[]), patch(
        "app.retrieval.retriever.search_chunks_by_full_text", return_value=[]
    ), patch("app.retrieval.retriever.reciprocal_rank_fusion", return_value=fused_hits):
        result = retrieve_passages(
            session,
            "missing chunk",
            embed_query=lambda _query: [0.0] * 768,
        )

    assert result.passages == []

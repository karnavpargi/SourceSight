from __future__ import annotations

import uuid
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from app.database.document_chunk import DocumentChunk
from app.database.source_document import SourceDocument
from app.retrieval.document_retriever import (
    SessionDocumentRetriever,
    SessionPerCallDocumentRetriever,
)
from app.retrieval.retriever import DEFAULT_RETRIEVAL_LIMIT
from app.retrieval.types import RetrievalResult, SourcePassage

CHUNK_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
DOCUMENT_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


def _document() -> SourceDocument:
    return SourceDocument(
        id=DOCUMENT_ID,
        ticker="AMZN",
        cik="0001018724",
        company_name="Amazon.com, Inc.",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001018724-24-000123",
        filing_date=date(2024, 2, 2),
        report_date=date(2023, 12, 31),
        primary_document="amzn.htm",
        source_url="https://example.com/amzn-10k",
        markdown_content="# filing",
        metadata_={},
    )


def _chunk() -> DocumentChunk:
    chunk = DocumentChunk(
        id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        chunk_index=3,
        page=None,
        section="Item 7. MD&A",
        content="AWS operating income increased.",
        token_count=10,
        embedding=None,
        metadata_={},
    )
    document = _document()
    chunk.document = document
    return chunk


def _passage() -> SourcePassage:
    return SourcePassage(
        chunk_id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        chunk_index=3,
        content="AWS operating income increased.",
        section="Item 7. MD&A",
        ticker="AMZN",
        company_name="Amazon.com, Inc.",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001018724-24-000123",
        filing_date=date(2024, 2, 2),
        source_url="https://example.com/amzn-10k",
        score=0.0,
    )


def test_session_document_retriever_search_filings() -> None:
    session = MagicMock()
    expected = RetrievalResult(query="aws", passages=[_passage()])

    with patch(
        "app.retrieval.document_retriever.retrieve_passages",
        return_value=expected,
    ) as retrieve:
        result = SessionDocumentRetriever(session, neighbor_window=2).search_filings(
            "aws",
            limit=5,
        )

    assert result == expected
    retrieve.assert_called_once_with(session, "aws", limit=5, neighbor_window=2)


def test_session_document_retriever_read_chunk_returns_passage() -> None:
    session = MagicMock()
    chunk = _chunk()

    with patch(
        "app.retrieval.document_retriever._load_chunks",
        return_value={CHUNK_ID: chunk},
    ), patch(
        "app.retrieval.document_retriever._to_source_passage",
        return_value=_passage(),
    ) as to_passage:
        result = SessionDocumentRetriever(session).read_chunk(CHUNK_ID)

    assert result == _passage()
    to_passage.assert_called_once_with(chunk, score=0.0)


def test_session_document_retriever_read_chunk_raises_when_missing() -> None:
    session = MagicMock()

    with patch("app.retrieval.document_retriever._load_chunks", return_value={}):
        with pytest.raises(ValueError, match="not found"):
            SessionDocumentRetriever(session).read_chunk(CHUNK_ID)


def test_session_document_retriever_read_surrounding_chunks() -> None:
    session = MagicMock()
    chunk = _chunk()
    neighbors = [_passage()]

    with patch(
        "app.retrieval.document_retriever._load_chunks",
        return_value={CHUNK_ID: chunk},
    ), patch(
        "app.retrieval.document_retriever._to_source_passage",
        return_value=_passage(),
    ), patch(
        "app.retrieval.document_retriever._load_neighbor_passages",
        return_value=neighbors,
    ) as load_neighbors:
        result = SessionDocumentRetriever(session).read_surrounding_chunks(
            CHUNK_ID,
            window=2,
        )

    assert result == neighbors
    load_neighbors.assert_called_once_with(
        session,
        [_passage()],
        neighbor_window=2,
        existing_chunk_ids={CHUNK_ID},
    )


def test_session_document_retriever_read_surrounding_chunks_raises_when_missing() -> None:
    session = MagicMock()

    with patch("app.retrieval.document_retriever._load_chunks", return_value={}):
        with pytest.raises(ValueError, match="not found"):
            SessionDocumentRetriever(session).read_surrounding_chunks(CHUNK_ID)


def test_session_per_call_document_retriever_opens_session_per_method() -> None:
    session = MagicMock()
    expected = RetrievalResult(query="aws", passages=[])

    with patch(
        "app.retrieval.document_retriever.session_scope"
    ) as scope, patch.object(
        SessionDocumentRetriever,
        "search_filings",
        return_value=expected,
    ) as search, patch.object(
        SessionDocumentRetriever,
        "read_chunk",
        return_value=_passage(),
    ) as read_chunk, patch.object(
        SessionDocumentRetriever,
        "read_surrounding_chunks",
        return_value=[_passage()],
    ) as read_neighbors:
        scope.return_value.__enter__.return_value = session
        retriever = SessionPerCallDocumentRetriever(neighbor_window=3)

        assert retriever.search_filings("aws") == expected
        assert retriever.read_chunk(CHUNK_ID) == _passage()
        assert retriever.read_surrounding_chunks(CHUNK_ID, window=2) == [_passage()]

    assert scope.call_count == 3
    search.assert_called_once_with("aws", limit=DEFAULT_RETRIEVAL_LIMIT)
    read_chunk.assert_called_once_with(CHUNK_ID)
    read_neighbors.assert_called_once_with(CHUNK_ID, window=2)

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from app.database.document_chunk import DocumentChunk
from app.database.source_document import SourceDocument
from ingest.chunk import TextChunk
from ingest.load import filing_exists, load_filing


def test_filing_exists_true() -> None:
    session = MagicMock()
    session.execute.return_value.scalar_one_or_none.return_value = "doc-id"
    assert filing_exists(
        session,
        ticker="AAPL",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0000320193-24-000123",
    )


def test_filing_exists_false() -> None:
    session = MagicMock()
    session.execute.return_value.scalar_one_or_none.return_value = None
    assert not filing_exists(
        session,
        ticker="AAPL",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0000320193-24-000123",
    )


def test_load_filing_mismatched_counts() -> None:
    session = MagicMock()
    chunks = [TextChunk(0, "text", "Item 1", None, 1)]
    with pytest.raises(ValueError, match="chunk and embedding counts must match"):
        load_filing(
            session,
            ticker="AAPL",
            cik="0000320193",
            company_name="Apple Inc.",
            form_type="10-K",
            fiscal_year=2024,
            accession_number="0000320193-24-000123",
            filing_date=date(2024, 11, 1),
            report_date=date(2024, 9, 28),
            primary_document="aapl.htm",
            source_url="https://example.com",
            markdown_content="# Filing",
            chunks=chunks,
            embeddings=[],
        )


def test_load_filing_writes_document_and_chunks() -> None:
    session = MagicMock()

    chunks = [
        TextChunk(0, "Risk factors text.", "Item 1A. Risk Factors", None, 4),
    ]
    embeddings = [[0.1] * 768]

    doc_id = load_filing(
        session,
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0000320193-24-000123",
        filing_date=date(2024, 11, 1),
        report_date=date(2024, 9, 28),
        primary_document="aapl.htm",
        source_url="https://example.com",
        markdown_content="# Filing",
        chunks=chunks,
        embeddings=embeddings,
        extra_metadata={"local_path": "2024/aapl.htm"},
    )

    assert session.add.call_count == 2
    session.flush.assert_called_once()
    added_types = {type(call.args[0]) for call in session.add.call_args_list}
    assert SourceDocument in added_types
    assert DocumentChunk in added_types

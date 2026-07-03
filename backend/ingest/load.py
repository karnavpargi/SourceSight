"""Persist extracted filings and chunks to Postgres."""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.document_chunk import DocumentChunk
from app.database.source_document import SourceDocument
from ingest.chunk import TextChunk


def filing_exists(
    session: Session,
    *,
    ticker: str,
    form_type: str,
    fiscal_year: int,
    accession_number: str,
) -> bool:
    statement = select(SourceDocument.id).where(
        SourceDocument.ticker == ticker,
        SourceDocument.form_type == form_type,
        SourceDocument.fiscal_year == fiscal_year,
        SourceDocument.accession_number == accession_number,
    )
    return session.execute(statement).scalar_one_or_none() is not None


def load_filing(
    session: Session,
    *,
    ticker: str,
    cik: str | None,
    company_name: str | None,
    form_type: str,
    fiscal_year: int,
    accession_number: str,
    filing_date: date,
    report_date: date | None,
    primary_document: str | None,
    source_url: str,
    markdown_content: str,
    chunks: list[TextChunk],
    embeddings: list[list[float] | None],
    extra_metadata: dict | None = None,
) -> uuid.UUID:
    if len(chunks) != len(embeddings):
        raise ValueError("chunk and embedding counts must match")

    document = SourceDocument(
        ticker=ticker,
        cik=cik,
        company_name=company_name,
        form_type=form_type,
        fiscal_year=fiscal_year,
        accession_number=accession_number,
        filing_date=filing_date,
        report_date=report_date,
        primary_document=primary_document,
        source_url=source_url,
        markdown_content=markdown_content,
        metadata_=extra_metadata or {},
    )
    session.add(document)
    session.flush()

    chunk_metadata_base = {
        "ticker": ticker,
        "company_name": company_name,
        "form_type": form_type,
        "fiscal_year": fiscal_year,
        "accession_number": accession_number,
        "filing_date": filing_date.isoformat(),
    }
    if report_date is not None:
        chunk_metadata_base["report_date"] = report_date.isoformat()

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        metadata = {
            **chunk_metadata_base,
            "section": chunk.section,
        }
        session.add(
            DocumentChunk(
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                page=chunk.page,
                section=chunk.section,
                content=chunk.content,
                token_count=chunk.token_count,
                embedding=embedding,
                metadata_=metadata,
            )
        )

    return document.id

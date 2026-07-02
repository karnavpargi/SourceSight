"""Retrieval result types."""

from __future__ import annotations

import uuid
from datetime import date

from pydantic import BaseModel, Field


class SourcePassage(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    content: str
    section: str | None = None
    page: int | None = None
    ticker: str
    company_name: str | None = None
    form_type: str
    fiscal_year: int
    accession_number: str
    filing_date: date
    report_date: date | None = None
    source_url: str
    score: float
    is_neighbor: bool = False


class RetrievalResult(BaseModel):
    query: str
    passages: list[SourcePassage] = Field(default_factory=list)

"""Read-only corpus status helpers."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.document_chunk import DocumentChunk
from app.database.source_document import SourceDocument


def corpus_counts(session: Session) -> tuple[int, int]:
    """Return (document_count, chunk_count) for the indexed filing corpus."""
    document_count = session.scalar(select(func.count()).select_from(SourceDocument)) or 0
    chunk_count = session.scalar(select(func.count()).select_from(DocumentChunk)) or 0
    return int(document_count), int(chunk_count)

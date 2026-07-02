"""Concrete DocumentRetriever backed by a SQLAlchemy session."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.orm import Session

from app.database.session import session_scope
from app.retrieval.retriever import (
    DEFAULT_NEIGHBOR_WINDOW,
    DEFAULT_RETRIEVAL_LIMIT,
    _load_chunks,
    _load_neighbor_passages,
    _to_source_passage,
    retrieve_passages,
)
from app.retrieval.types import RetrievalResult, SourcePassage


@dataclass
class SessionDocumentRetriever:
    """DocumentRetriever implementation for one request-scoped database session."""

    session: Session
    neighbor_window: int = DEFAULT_NEIGHBOR_WINDOW

    def search_filings(self, query: str, *, limit: int = DEFAULT_RETRIEVAL_LIMIT) -> RetrievalResult:
        return retrieve_passages(
            self.session,
            query,
            limit=limit,
            neighbor_window=self.neighbor_window,
        )

    def read_chunk(self, chunk_id: UUID) -> SourcePassage:
        chunks = _load_chunks(self.session, [chunk_id])
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise ValueError(f"Chunk {chunk_id} not found.")
        return _to_source_passage(chunk, score=0.0)

    def read_surrounding_chunks(
        self,
        chunk_id: UUID,
        *,
        window: int = 1,
    ) -> list[SourcePassage]:
        chunks = _load_chunks(self.session, [chunk_id])
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise ValueError(f"Chunk {chunk_id} not found.")

        anchor = _to_source_passage(chunk, score=0.0)
        return _load_neighbor_passages(
            self.session,
            [anchor],
            neighbor_window=window,
            existing_chunk_ids={chunk_id},
        )


@dataclass
class SessionPerCallDocumentRetriever:
    """Opens a fresh DB session for each tool call.

    PydanticAI runs sync tools on worker threads, so a single shared session is
    not safe across concurrent tool invocations.
    """

    neighbor_window: int = DEFAULT_NEIGHBOR_WINDOW

    def search_filings(self, query: str, *, limit: int = DEFAULT_RETRIEVAL_LIMIT) -> RetrievalResult:
        with session_scope() as session:
            return SessionDocumentRetriever(
                session,
                neighbor_window=self.neighbor_window,
            ).search_filings(query, limit=limit)

    def read_chunk(self, chunk_id: UUID) -> SourcePassage:
        with session_scope() as session:
            return SessionDocumentRetriever(
                session,
                neighbor_window=self.neighbor_window,
            ).read_chunk(chunk_id)

    def read_surrounding_chunks(
        self,
        chunk_id: UUID,
        *,
        window: int = 1,
    ) -> list[SourcePassage]:
        with session_scope() as session:
            return SessionDocumentRetriever(
                session,
                neighbor_window=self.neighbor_window,
            ).read_surrounding_chunks(chunk_id, window=window)

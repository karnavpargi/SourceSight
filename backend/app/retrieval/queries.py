"""Ranked chunk search queries for hybrid retrieval."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.database.document_chunk import DocumentChunk
from app.database.source_document import SourceDocument
from app.retrieval.fts_query import fts_query_variants

DEFAULT_SEARCH_LIMIT = 20


@dataclass(frozen=True)
class RankedChunkHit:
    chunk_id: uuid.UUID
    score: float


def search_chunks_by_embedding(
    session: Session,
    embedding: list[float],
    *,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[RankedChunkHit]:
    """Return chunks ranked by pgvector cosine similarity (higher score is better)."""
    if limit <= 0:
        return []
    if len(embedding) != settings.embedding_dimensions:
        raise ValueError(
            f"Query embedding has {len(embedding)} dimensions; "
            f"expected {settings.embedding_dimensions}."
        )

    distance = DocumentChunk.embedding.cosine_distance(embedding)
    statement = (
        select(
            DocumentChunk.id,
            distance.label("distance"),
        )
        .where(DocumentChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(limit)
    )
    return _rows_to_hits(session.execute(statement), score_from_distance=True)


def search_chunks_by_full_text(
    session: Session,
    query: str,
    *,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> list[RankedChunkHit]:
    """Return chunks ranked by Postgres full-text search (higher score is better)."""
    if limit <= 0:
        return []

    normalized_query = query.strip()
    if not normalized_query:
        return []

    for fts_query, tickers in fts_query_variants(normalized_query):
        hits = _full_text_search(
            session,
            fts_query,
            tickers=tickers,
            limit=limit,
        )
        if hits:
            return hits
    return []


def _full_text_search(
    session: Session,
    fts_query: str,
    *,
    tickers: list[str],
    limit: int,
) -> list[RankedChunkHit]:
    ts_query = func.websearch_to_tsquery("english", fts_query)
    rank = func.ts_rank_cd(DocumentChunk.search_vector, ts_query)
    statement = (
        select(
            DocumentChunk.id,
            rank.label("score"),
        )
        .where(DocumentChunk.search_vector.op("@@")(ts_query))
        .order_by(rank.desc(), DocumentChunk.id)
        .limit(limit)
    )
    if tickers:
        statement = statement.join(
            SourceDocument,
            SourceDocument.id == DocumentChunk.document_id,
        ).where(SourceDocument.ticker.in_(tickers))
    return _rows_to_hits(session.execute(statement), score_from_distance=False)


def _rows_to_hits(result, *, score_from_distance: bool) -> list[RankedChunkHit]:
    hits: list[RankedChunkHit] = []
    for row in result.all():
        raw_score = float(row.score if not score_from_distance else row.distance)
        score = 1.0 - raw_score if score_from_distance else raw_score
        hits.append(RankedChunkHit(chunk_id=row.id, score=score))
    return hits


def build_embedding_search_statement(
    embedding: list[float],
    *,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> Select[tuple[uuid.UUID, float]]:
    """Expose the SQLAlchemy select for tests and query inspection."""
    distance = DocumentChunk.embedding.cosine_distance(embedding)
    return (
        select(
            DocumentChunk.id,
            distance.label("distance"),
        )
        .where(DocumentChunk.embedding.is_not(None))
        .order_by(distance)
        .limit(limit)
    )


def build_full_text_search_statement(
    query: str,
    *,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> Select[tuple[uuid.UUID, float]]:
    """Expose the SQLAlchemy select for tests and query inspection."""
    ts_query = func.websearch_to_tsquery("english", query.strip())
    rank = func.ts_rank_cd(DocumentChunk.search_vector, ts_query)
    return (
        select(
            DocumentChunk.id,
            rank.label("score"),
        )
        .where(DocumentChunk.search_vector.op("@@")(ts_query))
        .order_by(rank.desc(), DocumentChunk.id)
        .limit(limit)
    )

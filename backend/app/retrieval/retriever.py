"""Hybrid retrieval orchestrator."""

from __future__ import annotations

import uuid
from collections.abc import Callable

import httpx
import structlog
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.config import settings
from app.database.document_chunk import DocumentChunk
from app.database.source_document import SourceDocument
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.queries import (
    DEFAULT_SEARCH_LIMIT,
    RankedChunkHit,
    search_chunks_by_embedding,
    search_chunks_by_full_text,
)
from app.retrieval.types import RetrievalResult, SourcePassage
from ingest.embed import embed_texts

DEFAULT_RETRIEVAL_LIMIT = 10
DEFAULT_NEIGHBOR_WINDOW = 0

EmbedQueryFn = Callable[[str], list[float]]

logger = structlog.get_logger(__name__)


def retrieve_passages(
    session: Session,
    query: str,
    *,
    limit: int = DEFAULT_RETRIEVAL_LIMIT,
    candidate_limit: int = DEFAULT_SEARCH_LIMIT,
    neighbor_window: int = DEFAULT_NEIGHBOR_WINDOW,
    embed_query: EmbedQueryFn | None = None,
) -> RetrievalResult:
    """Embed the query, run hybrid search, fuse, and load passage metadata."""
    normalized_query = query.strip()
    if not normalized_query:
        return RetrievalResult(query=query, passages=[])

    embed_fn = embed_query or _default_embed_query
    vector_hits = _vector_search_hits(
        session,
        embed_fn,
        normalized_query,
        candidate_limit=candidate_limit,
    )
    text_hits = search_chunks_by_full_text(
        session,
        normalized_query,
        limit=candidate_limit,
    )
    fused_hits = reciprocal_rank_fusion(vector_hits, text_hits, limit=limit)
    if not fused_hits:
        return RetrievalResult(query=normalized_query, passages=[])

    score_by_chunk_id = {hit.chunk_id: hit.score for hit in fused_hits}
    ranked_chunk_ids = [hit.chunk_id for hit in fused_hits]
    chunks = _load_chunks(session, ranked_chunk_ids)
    passages = [
        _to_source_passage(chunks[chunk_id], score=score_by_chunk_id[chunk_id])
        for chunk_id in ranked_chunk_ids
        if chunk_id in chunks
    ]

    if neighbor_window > 0 and passages:
        neighbor_passages = _load_neighbor_passages(
            session,
            passages,
            neighbor_window=neighbor_window,
            existing_chunk_ids=set(chunks),
        )
        passages.extend(neighbor_passages)

    return RetrievalResult(query=normalized_query, passages=passages)


def _vector_search_hits(
    session: Session,
    embed_fn: EmbedQueryFn,
    query: str,
    *,
    candidate_limit: int,
) -> list[RankedChunkHit]:
    try:
        query_embedding = embed_fn(query)
    except httpx.HTTPError:
        logger.warning(
            "retrieval.embedding_unavailable",
            ollama_base_url=settings.ollama_base_url,
        )
        return []

    return search_chunks_by_embedding(
        session,
        query_embedding,
        limit=candidate_limit,
    )


def _default_embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


def _load_chunks(
    session: Session,
    chunk_ids: list[uuid.UUID],
) -> dict[uuid.UUID, DocumentChunk]:
    if not chunk_ids:
        return {}

    statement = (
        select(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .where(DocumentChunk.id.in_(chunk_ids))
    )
    rows = session.execute(statement).scalars().all()
    return {chunk.id: chunk for chunk in rows}


def _load_neighbor_passages(
    session: Session,
    passages: list[SourcePassage],
    *,
    neighbor_window: int,
    existing_chunk_ids: set[uuid.UUID],
) -> list[SourcePassage]:
    neighbor_specs: set[tuple[uuid.UUID, int]] = set()
    for passage in passages:
        for offset in range(-neighbor_window, neighbor_window + 1):
            if offset == 0:
                continue
            neighbor_specs.add((passage.document_id, passage.chunk_index + offset))

    if not neighbor_specs:
        return []

    document_ids = {document_id for document_id, _ in neighbor_specs}
    statement = (
        select(DocumentChunk)
        .options(joinedload(DocumentChunk.document))
        .where(DocumentChunk.document_id.in_(document_ids))
    )
    rows = session.execute(statement).scalars().all()

    neighbors: list[SourcePassage] = []
    for chunk in rows:
        if chunk.id in existing_chunk_ids:
            continue
        if (chunk.document_id, chunk.chunk_index) not in neighbor_specs:
            continue
        neighbors.append(_to_source_passage(chunk, score=0.0, is_neighbor=True))
        existing_chunk_ids.add(chunk.id)

    neighbors.sort(key=lambda passage: (passage.document_id, passage.chunk_index))
    return neighbors


def _to_source_passage(
    chunk: DocumentChunk,
    *,
    score: float,
    is_neighbor: bool = False,
) -> SourcePassage:
    document = chunk.document
    if document is None:
        raise ValueError(f"Chunk {chunk.id} is missing its source document.")

    return SourcePassage(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        section=chunk.section,
        page=chunk.page,
        ticker=document.ticker,
        company_name=document.company_name,
        form_type=document.form_type,
        fiscal_year=document.fiscal_year,
        accession_number=document.accession_number,
        filing_date=document.filing_date,
        report_date=document.report_date,
        source_url=document.source_url,
        score=score,
        is_neighbor=is_neighbor,
    )

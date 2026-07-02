"""Reciprocal Rank Fusion for hybrid retrieval."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from app.retrieval.queries import RankedChunkHit

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedChunkHit:
    chunk_id: uuid.UUID
    score: float


def reciprocal_rank_fusion(
    *ranked_lists: Sequence[RankedChunkHit] | Sequence[uuid.UUID],
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
) -> list[FusedChunkHit]:
    """Fuse multiple ranked chunk lists with Reciprocal Rank Fusion."""
    if k <= 0:
        raise ValueError("RRF constant k must be positive.")
    if not ranked_lists:
        return []

    scores: dict[uuid.UUID, float] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(_chunk_ids(ranked_list), start=1):
            chunk_id = item
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    fused = [
        FusedChunkHit(chunk_id=chunk_id, score=score)
        for chunk_id, score in sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
    ]
    if limit is not None:
        if limit <= 0:
            return []
        return fused[:limit]
    return fused


def _chunk_ids(ranked_list: Sequence[RankedChunkHit] | Sequence[uuid.UUID]) -> list[uuid.UUID]:
    if not ranked_list:
        return []
    first = ranked_list[0]
    if isinstance(first, RankedChunkHit):
        return [hit.chunk_id for hit in ranked_list]  # type: ignore[misc]
    return list(ranked_list)  # type: ignore[arg-type]

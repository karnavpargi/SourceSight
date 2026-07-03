from __future__ import annotations

import uuid

import pytest

from app.retrieval.fusion import DEFAULT_RRF_K, FusedChunkHit, reciprocal_rank_fusion
from app.retrieval.queries import RankedChunkHit


def test_reciprocal_rank_fusion_orders_chunks_present_in_both_lists_higher() -> None:
    doc_a = uuid.UUID("00000000-0000-0000-0000-000000000001")
    doc_b = uuid.UUID("00000000-0000-0000-0000-000000000002")
    doc_c = uuid.UUID("00000000-0000-0000-0000-000000000003")

    vector_hits = [
        RankedChunkHit(chunk_id=doc_a, score=0.9),
        RankedChunkHit(chunk_id=doc_b, score=0.8),
        RankedChunkHit(chunk_id=doc_c, score=0.7),
    ]
    text_hits = [
        RankedChunkHit(chunk_id=doc_b, score=0.5),
        RankedChunkHit(chunk_id=doc_c, score=0.4),
        RankedChunkHit(chunk_id=doc_a, score=0.3),
    ]

    fused = reciprocal_rank_fusion(vector_hits, text_hits)

    assert [hit.chunk_id for hit in fused] == [doc_b, doc_a, doc_c]
    assert fused[0].score > fused[1].score > fused[2].score


def test_reciprocal_rank_fusion_uses_expected_rrf_scores() -> None:
    doc_a = uuid.UUID("00000000-0000-0000-0000-00000000000a")
    doc_b = uuid.UUID("00000000-0000-0000-0000-00000000000b")

    fused = reciprocal_rank_fusion([doc_a, doc_b], [doc_b], k=60)

    expected_a = 1.0 / 61.0
    expected_b = (1.0 / 62.0) + (1.0 / 61.0)
    assert fused == [
        FusedChunkHit(chunk_id=doc_b, score=expected_b),
        FusedChunkHit(chunk_id=doc_a, score=expected_a),
    ]


def test_reciprocal_rank_fusion_accepts_chunk_id_lists() -> None:
    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()

    fused = reciprocal_rank_fusion([doc_a], [doc_b])

    assert {hit.chunk_id for hit in fused} == {doc_a, doc_b}
    assert all(hit.score == pytest.approx(1.0 / 61.0) for hit in fused)


def test_reciprocal_rank_fusion_respects_limit() -> None:
    ids = [uuid.uuid4() for _ in range(5)]
    fused = reciprocal_rank_fusion(ids, limit=2)
    assert len(fused) == 2


def test_reciprocal_rank_fusion_returns_empty_for_no_lists() -> None:
    assert reciprocal_rank_fusion() == []


def test_reciprocal_rank_fusion_returns_empty_for_zero_limit() -> None:
    doc_id = uuid.uuid4()
    assert reciprocal_rank_fusion([doc_id], limit=0) == []


def test_reciprocal_rank_fusion_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion([uuid.uuid4()], k=0)


def test_reciprocal_rank_fusion_skips_empty_ranked_lists() -> None:
    doc_id = uuid.uuid4()
    fused = reciprocal_rank_fusion([doc_id], [])
    assert fused == [FusedChunkHit(chunk_id=doc_id, score=pytest.approx(1.0 / 61.0))]


def test_default_rrf_k_is_sixty() -> None:
    assert DEFAULT_RRF_K == 60

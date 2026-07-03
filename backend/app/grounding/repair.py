"""Best-effort repair of common LLM grounding output gaps."""

from __future__ import annotations

import re

from uuid import UUID

from app.assistant.outputs import Citation, GroundedAnswer
from app.retrieval.types import SourcePassage

CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")
EXCERPT_MAX_LEN = 240


def repair_grounded_answer(
    answer: GroundedAnswer,
    retrieved_passages: list[SourcePassage],
) -> GroundedAnswer:
    """Fill missing citation records when markers or cited_passages are present."""
    markers = sorted({int(match) for match in CITATION_MARKER_RE.findall(answer.answer)})
    if not markers:
        return answer

    passages_by_id = {passage.chunk_id: passage for passage in retrieved_passages}
    for passage in answer.cited_passages:
        passages_by_id.setdefault(passage.chunk_id, passage)

    existing = {citation.citation_index: citation for citation in answer.citations}
    if set(existing) >= set(markers):
        return answer

    repaired = list(answer.citations)
    for marker in markers:
        if marker in existing:
            continue
        chunk_id = _chunk_id_for_marker(
            marker,
            cited_passages=answer.cited_passages,
            retrieved_passages=retrieved_passages,
            passages_by_id=passages_by_id,
        )
        if chunk_id is None:
            continue
        repaired.append(
            Citation(
                citation_index=marker,
                chunk_id=chunk_id,
                excerpt=_default_excerpt(passages_by_id[chunk_id]),
            )
        )

    if len(repaired) == len(answer.citations):
        return answer

    repaired.sort(key=lambda citation: citation.citation_index)
    cited_passages = list(answer.cited_passages)
    cited_ids = {passage.chunk_id for passage in cited_passages}
    for citation in repaired:
        if citation.chunk_id in cited_ids:
            continue
        passage = passages_by_id.get(citation.chunk_id)
        if passage is not None:
            cited_passages.append(passage)
            cited_ids.add(passage.chunk_id)

    return answer.model_copy(
        update={
            "citations": repaired,
            "cited_passages": cited_passages,
        }
    )


def _chunk_id_for_marker(
    marker: int,
    *,
    cited_passages: list[SourcePassage],
    retrieved_passages: list[SourcePassage],
    passages_by_id: dict[UUID, SourcePassage],
) -> UUID | None:
    if marker <= len(cited_passages):
        candidate = cited_passages[marker - 1].chunk_id
        if candidate in passages_by_id:
            return candidate
    if marker <= len(retrieved_passages):
        candidate = retrieved_passages[marker - 1].chunk_id
        if candidate in passages_by_id:
            return candidate
    return None


def _default_excerpt(passage: SourcePassage) -> str:
    text = passage.content.strip()
    if len(text) <= EXCERPT_MAX_LEN:
        return text
    return f"{text[: EXCERPT_MAX_LEN - 3].rstrip()}..."

"""Best-effort repair of common LLM grounding output gaps."""

from __future__ import annotations

import re

from uuid import UUID

from app.assistant.outputs import Citation, GroundedAnswer
from app.grounding.validator import (
    CITATION_MARKER_RE,
    _claim_segments,
    _uncited_factual_segments,
)
from app.retrieval.types import SourcePassage

EXCERPT_MAX_LEN = 240
_TERMINAL_PUNCT_RE = re.compile(r"^(.*?)([.!?])?\s*$", re.DOTALL)


def repair_grounded_answer(
    answer: GroundedAnswer,
    retrieved_passages: list[SourcePassage],
) -> GroundedAnswer:
    """Fill missing citation records / inline markers for common model slips."""
    repaired = _repair_missing_citation_records(answer, retrieved_passages)
    return _repair_inline_citation_markers(repaired)


def _repair_missing_citation_records(
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


def _repair_inline_citation_markers(answer: GroundedAnswer) -> GroundedAnswer:
    """Inject missing inline markers into claim sentences for orphan records."""
    if not answer.citations:
        return answer

    citation_indices = sorted(
        {citation.citation_index for citation in answer.citations}
    )
    text = answer.answer
    markers = {int(match) for match in CITATION_MARKER_RE.findall(text)}
    orphans = [index for index in citation_indices if index not in markers]
    uncited_segments = _uncited_factual_segments(text)

    if not orphans and not uncited_segments:
        return answer

    for segment in uncited_segments:
        if orphans:
            marker = orphans.pop(0)
        else:
            marker = citation_indices[0]
        cited_segment = _inject_marker_before_terminal_punct(segment, marker)
        text = text.replace(segment, cited_segment, 1)

    if orphans:
        text = _inject_markers_into_last_segment(text, orphans)

    if text == answer.answer:
        return answer
    return answer.model_copy(update={"answer": text})


def _inject_marker_before_terminal_punct(segment: str, marker: int) -> str:
    match = _TERMINAL_PUNCT_RE.match(segment.strip())
    if match is None:
        return f"{segment.rstrip()} [{marker}]"
    body = match.group(1).rstrip()
    punct = match.group(2) or ""
    return f"{body} [{marker}]{punct}"


def _inject_markers_into_last_segment(text: str, markers: list[int]) -> str:
    segments = _claim_segments(text)
    if not segments:
        suffix = "".join(f" [{index}]" for index in markers)
        return f"{text.rstrip()}{suffix}"

    last = segments[-1]
    cited = last
    for marker in markers:
        cited = _inject_marker_before_terminal_punct(cited, marker)
    return text.replace(last, cited, 1)


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

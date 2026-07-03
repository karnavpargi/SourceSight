"""Grounding checks for assistant answers."""

from __future__ import annotations

import re

from app.assistant.outputs import GroundedAnswer
from app.retrieval.types import SourcePassage

CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
REFUSAL_PREFIX = "this corpus does not contain enough evidence"


class GroundingError(Exception):
    """Raised when an answer fails grounding policy checks."""


def _normalize_refusal_text(text: str) -> str:
    normalized = text.strip().lower().replace("doesn't", "does not")
    return re.sub(r"\s+", " ", normalized)


def _is_refusal(answer: str) -> bool:
    return _normalize_refusal_text(answer).startswith(REFUSAL_PREFIX)


def _citation_markers_in_answer(answer: str) -> set[int]:
    return {int(match) for match in CITATION_MARKER_RE.findall(answer)}


def _claim_segments(answer: str) -> list[str]:
    segments: list[str] = []
    for paragraph in answer.splitlines():
        for sentence in SENTENCE_SPLIT_RE.split(paragraph.strip()):
            segment = sentence.strip()
            if segment:
                segments.append(segment)
    return segments


def validate(answer: GroundedAnswer, retrieved_passages: list[SourcePassage]) -> None:
    """Ensure grounded answers cite only retrieved evidence."""
    if _is_refusal(answer.answer):
        if answer.citations:
            raise GroundingError("Refusal answers must not include citations.")
        return

    if not answer.citations:
        raise GroundingError("Grounded answers must include at least one citation.")

    markers_in_answer = _citation_markers_in_answer(answer.answer)
    citation_indices = {citation.citation_index for citation in answer.citations}

    unresolved_markers = markers_in_answer - citation_indices
    if unresolved_markers:
        missing = ", ".join(f"[{index}]" for index in sorted(unresolved_markers))
        raise GroundingError(f"Answer cites {missing} without matching citation records.")

    orphan_citations = citation_indices - markers_in_answer
    if orphan_citations:
        missing = ", ".join(f"[{index}]" for index in sorted(orphan_citations))
        raise GroundingError(f"Citation records exist for {missing} but the answer text omits them.")

    uncited_segments = [
        segment
        for segment in _claim_segments(answer.answer)
        if not CITATION_MARKER_RE.search(segment)
    ]
    if uncited_segments:
        preview = uncited_segments[0][:120]
        raise GroundingError(
            "Every factual claim must include at least one citation marker. "
            f"Uncited segment: {preview!r}"
        )

    retrieved_chunk_ids = {passage.chunk_id for passage in retrieved_passages}
    for citation in answer.citations:
        if citation.chunk_id not in retrieved_chunk_ids:
            raise GroundingError(
                f"Citation [{citation.citation_index}] references chunk "
                f"{citation.chunk_id} that was not retrieved for this request."
            )


class GroundingValidatorService:
    """GroundingValidator protocol implementation for agent deps."""

    def validate(
        self,
        answer: GroundedAnswer,
        retrieved_passages: list[SourcePassage],
    ) -> None:
        validate(answer, retrieved_passages)


grounding_validator = GroundingValidatorService()

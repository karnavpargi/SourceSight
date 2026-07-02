"""Grounding checks for assistant answers."""

from __future__ import annotations

from app.assistant.outputs import GroundedAnswer
from app.retrieval.types import SourcePassage


class GroundingError(Exception):
    """Raised when an answer fails grounding policy checks."""


def validate(answer: GroundedAnswer, retrieved_passages: list[SourcePassage]) -> None:
    """Ensure citations only reference chunks retrieved during this turn."""
    if not answer.citations:
        return

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

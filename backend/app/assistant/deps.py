"""Runtime dependencies injected into the document assistant agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.assistant.outputs import GroundedAnswer
from app.retrieval.types import RetrievalResult, SourcePassage

__all__ = ["DocumentAgentDeps", "DocumentRetriever", "GroundingValidator"]


class DocumentRetriever(Protocol):
    """Bounded retrieval interface exposed to the agent tools."""

    def search_filings(self, query: str, *, limit: int = 10) -> RetrievalResult:
        """Run hybrid retrieval over the filing corpus."""
        ...

    def read_chunk(self, chunk_id: UUID) -> SourcePassage:
        """Load one chunk with source document metadata."""
        ...

    def read_surrounding_chunks(
        self,
        chunk_id: UUID,
        *,
        window: int = 1,
    ) -> list[SourcePassage]:
        """Load neighboring chunks around the target passage."""
        ...


class GroundingValidator(Protocol):
    """Checks that an answer cites retrieved evidence."""

    def validate(
        self,
        answer: GroundedAnswer,
        retrieved_passages: list[SourcePassage],
    ) -> None:
        """Raise if the answer fails grounding policy checks."""
        ...


@dataclass
class DocumentAgentDeps:
    """Per-turn dependencies for the PydanticAI document agent."""

    user_id: UUID
    thread_id: UUID
    retriever: DocumentRetriever
    grounding_validator: GroundingValidator

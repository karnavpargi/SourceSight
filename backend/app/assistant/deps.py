"""Runtime dependencies injected into the document assistant agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.assistant.evidence import EvidenceRegistry
from app.assistant.outputs import GroundedAnswer
from app.chat.turn_budget import DEFAULT_TURN_BUDGET, TurnBudget
from app.chat.usage import TurnUsage
from app.retrieval.types import RetrievalResult, SourcePassage

__all__ = ["DocumentAgentDeps", "DocumentRetriever", "GroundingValidator"]


class DocumentRetriever(Protocol):
    """Bounded retrieval interface exposed to the agent tools."""

    def search_filings(self, query: str, *, limit: int = 10) -> RetrievalResult:
        """Run hybrid retrieval over the filing corpus."""
        ...

    def search_filings_batch(
        self,
        queries: list[str],
        *,
        limit_per_query: int = 5,
    ) -> list[SourcePassage]:
        """Run batched hybrid retrieval over the filing corpus."""
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
    evidence: EvidenceRegistry
    usage: TurnUsage
    budget: TurnBudget = DEFAULT_TURN_BUDGET
    # Total cleaned search queries issued this turn.
    search_count: int = 0
    # When True, the agent is running a correction-only pass and must not
    # perform additional retrieval (search_filings should no-op).
    correction_mode: bool = False

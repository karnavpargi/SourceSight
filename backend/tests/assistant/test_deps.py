from dataclasses import dataclass
from uuid import UUID

from app.assistant.deps import DocumentAgentDeps
from app.assistant.evidence import EvidenceRegistry
from app.assistant.outputs import GroundedAnswer
from app.chat.turn_budget import DEFAULT_TURN_BUDGET, TurnBudget
from app.chat.usage import TurnUsage
from app.retrieval.types import RetrievalResult, SourcePassage

USER_ID = UUID("550e8400-e29b-41d4-a716-446655440000")
THREAD_ID = UUID("770e8400-e29b-41d4-a716-446655440002")
CHUNK_ID = UUID("11111111-1111-1111-1111-111111111111")


@dataclass
class StubRetriever:
    last_query: str | None = None

    def search_filings(self, query: str, *, limit: int = 10) -> RetrievalResult:
        self.last_query = query
        return RetrievalResult(query=query, passages=[])

    def search_filings_batch(
        self,
        queries: list[str],
        *,
        limit_per_query: int = 5,
    ) -> list[SourcePassage]:
        # Simple batch stub: record the joined query for assertions if needed.
        self.last_query = " | ".join(queries)
        return []

    def read_chunk(self, chunk_id: UUID) -> SourcePassage:
        raise NotImplementedError(chunk_id)

    def read_surrounding_chunks(
        self,
        chunk_id: UUID,
        *,
        window: int = 1,
    ) -> list[SourcePassage]:
        raise NotImplementedError((chunk_id, window))


@dataclass
class StubValidator:
    validated: bool = False

    def validate(
        self,
        answer: GroundedAnswer,
        retrieved_passages: list[SourcePassage],
    ) -> None:
        self.validated = True
        assert answer.answer
        assert retrieved_passages is not None


def test_document_agent_deps_wires_runtime_services() -> None:
    retriever = StubRetriever()
    validator = StubValidator()
    evidence = EvidenceRegistry()
    usage = TurnUsage()
    budget = DEFAULT_TURN_BUDGET

    deps = DocumentAgentDeps(
        user_id=USER_ID,
        thread_id=THREAD_ID,
        retriever=retriever,
        grounding_validator=validator,
        evidence=evidence,
        usage=usage,
        budget=budget,
    )

    result = deps.retriever.search_filings("AWS operating income")

    assert deps.user_id == USER_ID
    assert deps.thread_id == THREAD_ID
    assert retriever.last_query == "AWS operating income"
    assert result.passages == []
    assert deps.evidence is evidence
    assert deps.usage is usage
    assert deps.budget == budget
    assert deps.search_count == 0

    deps.grounding_validator.validate(
        GroundedAnswer(answer="No evidence in corpus."),
        [],
    )
    assert validator.validated is True

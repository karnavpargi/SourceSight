import json
from datetime import date
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart, models
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.assistant.agent import (
    _search_filings_impl,
    build_document_agent_model,
    chat_model_name,
    document_agent,
    load_instructions,
)
from app.assistant.deps import DocumentAgentDeps
from app.assistant.evidence import CompactEvidence, EvidenceRegistry
from app.assistant.outputs import GroundedDraft
from app.chat.turn_budget import DEFAULT_TURN_BUDGET
from app.chat.usage import TurnUsage
from app.retrieval.types import SourcePassage
from tests.assistant.test_deps import StubRetriever, StubValidator, THREAD_ID, USER_ID

models.ALLOW_MODEL_REQUESTS = False


def test_load_instructions_encodes_grounding_contract() -> None:
    instructions = load_instructions()

    assert "GroundedDraft" in instructions
    assert "search_filings" in instructions
    assert "queries" in instructions
    assert "E1" in instructions
    assert "read_chunk" not in instructions
    assert "read_surrounding_chunks" not in instructions
    assert "stock recommendation" in instructions.lower()


def test_document_agent_registers_retrieval_tools() -> None:
    tool_names = sorted(document_agent._function_toolset.tools.keys())

    assert tool_names == ["search_filings"]


def test_chat_model_name_uses_google_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.assistant.agent.settings.google_api_key", "test-google-key")

    assert chat_model_name("google", "gemini-2.0-flash") == "google:gemini-2.0-flash"

    model = build_document_agent_model("google", "gemini-2.0-flash")
    assert model.model_name == "gemini-2.0-flash"


def test_chat_model_name_uses_opencode_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.assistant.agent.settings.opencode_api_key", "test-opencode-key")

    assert chat_model_name("opencode", "glm-5.2") == "opencode:glm-5.2"

    model = build_document_agent_model("opencode", "glm-5.2")
    assert model.model_name == "glm-5.2"


@pytest.mark.anyio
async def test_document_agent_run_invokes_search_filings_tool() -> None:
    retriever = StubRetriever()
    deps = DocumentAgentDeps(
        user_id=USER_ID,
        thread_id=THREAD_ID,
        retriever=retriever,
        grounding_validator=StubValidator(),
        evidence=EvidenceRegistry(),
        usage=TurnUsage(),
        budget=DEFAULT_TURN_BUDGET,
    )
    step = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal step
        step += 1
        if step == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "search_filings",
                        {"queries": ["AMZN AWS operating income 2024"]},
                    )
                ]
            )

        return ModelResponse(
            parts=[
                TextPart(
                    json.dumps(
                        {
                            "answer": "This corpus does not contain enough evidence to answer that.",
                            "citations": [],
                        }
                    )
                )
            ]
        )

    with document_agent.override(model=FunctionModel(model_fn)):
        result = await document_agent.run("How did AWS operating income change?", deps=deps)

    assert retriever.last_query == "AMZN AWS operating income 2024"
    assert result.output == GroundedDraft(
        answer="This corpus does not contain enough evidence to answer that.",
        citations=[],
    )


def _sample_passage() -> SourcePassage:
    return SourcePassage(
        chunk_id=UUID("11111111-1111-1111-1111-111111111111"),
        document_id=UUID("22222222-2222-2222-2222-222222222222"),
        chunk_index=0,
        content="Sample passage content about AWS operating income.",
        section="Item 1. Business",
        ticker="AMZN",
        company_name="Amazon.com, Inc.",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001234567-24-000001",
        filing_date=date(2024, 2, 15),
        source_url="https://example.com/amzn-10k",
        score=0.9,
    )


def test_search_filings_tool_returns_compact_aliases_without_chunk_id() -> None:
    evidence = EvidenceRegistry()
    usage = TurnUsage()

    class StubBatchRetriever:
        def __init__(self) -> None:
            self.queries: list[list[str]] = []

        def search_filings_batch(
            self,
            queries: list[str],
            *,
            limit_per_query: int = 5,
        ) -> list[SourcePassage]:
            self.queries.append(queries)
            return [_sample_passage()]

    retriever = StubBatchRetriever()
    deps = SimpleNamespace(
        budget=DEFAULT_TURN_BUDGET,
        evidence=evidence,
        usage=usage,
        retriever=retriever,
        search_count=0,
        correction_mode=False,
    )

    result = _search_filings_impl(deps, ["AMZN AWS operating income 2024"])

    assert isinstance(result, list)
    assert all(isinstance(item, CompactEvidence) for item in result)

    dumped = [item.model_dump() for item in result]
    assert all("chunk_id" not in row for row in dumped)
    assert any(row["alias"].startswith("E") for row in dumped)
    # Embedding usage is approximated as one call per cleaned query when embeddings are enabled.
    assert usage.embedding_calls == 1


def test_search_filings_respects_query_budget_across_turn() -> None:
    evidence = EvidenceRegistry()
    usage = TurnUsage()

    class RecordingBatchRetriever:
        def __init__(self) -> None:
            self.queries: list[list[str]] = []

        def search_filings_batch(
            self,
            queries: list[str],
            *,
            limit_per_query: int = 5,
        ) -> list[SourcePassage]:
            self.queries.append(queries)
            return [_sample_passage()]

    retriever = RecordingBatchRetriever()
    deps = SimpleNamespace(
        budget=DEFAULT_TURN_BUDGET,
        evidence=evidence,
        usage=usage,
        retriever=retriever,
        search_count=0,
        correction_mode=False,
    )

    # DEFAULT_TURN_BUDGET.max_searches == 3; total cleaned queries across calls
    # must not exceed this value.
    _search_filings_impl(deps, ["q1", "q2"])
    _search_filings_impl(deps, ["q3", "q4"])
    third = _search_filings_impl(deps, ["q5"])

    # First call uses both queries, second call is capped to one remaining query,
    # and the third call is a no-op.
    assert retriever.queries == [["q1", "q2"], ["q3"]]
    assert third == []
    assert deps.search_count == DEFAULT_TURN_BUDGET.max_searches


def test_search_filings_noops_during_correction_mode() -> None:
    evidence = EvidenceRegistry()
    usage = TurnUsage()

    class FailingBatchRetriever:
        def search_filings_batch(
            self,
            queries: list[str],
            *,
            limit_per_query: int = 5,
        ) -> list[SourcePassage]:
            raise AssertionError("retriever should not be called during correction")

    retriever = FailingBatchRetriever()
    deps = SimpleNamespace(
        budget=DEFAULT_TURN_BUDGET,
        evidence=evidence,
        usage=usage,
        retriever=retriever,
        search_count=0,
        correction_mode=True,
    )

    result = _search_filings_impl(deps, ["AMZN AWS operating income 2024"])
    assert result == []


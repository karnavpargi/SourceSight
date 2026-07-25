from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from pydantic_ai import models
from pydantic_ai.exceptions import ModelHTTPError

from app.assistant.evidence import EvidenceRegistry
from app.assistant.facts import ExtractedFact, FactExtraction
from app.assistant.outputs import Citation, DraftCitation, GroundedAnswer, GroundedDraft
from app.chat.generation import ChatGenerationConfig
from app.chat.models_catalog import ResolvedChatModel
from app.chat.routing import QueryPlan
from app.chat.orchestrator import (
    REFUSAL_MESSAGE,
    model_unavailable_message,
    run_chat_turn,
    _run_routed_turn,
)
from app.chat.turn_budget import DEFAULT_TURN_BUDGET
from app.grounding.validator import GroundingError, grounding_validator
from app.database.chats import ChatMessageRecord
from app.chat.usage import TurnUsage
from app.retrieval.types import RetrievalResult, SourcePassage
from app.retrieval.coverage import CorpusCoverage

models.ALLOW_MODEL_REQUESTS = False

USER_ID = UUID("550e8400-e29b-41d4-a716-446655440000")
THREAD_ID = UUID("770e8400-e29b-41d4-a716-446655440002")
MESSAGE_ID = UUID("880e8400-e29b-41d4-a716-446655440003")
CHUNK_ID = UUID("11111111-1111-1111-1111-111111111111")
DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
TEST_CHAT_MODEL = ResolvedChatModel(provider="google", model="gemini-2.0-flash")
TEST_GENERATION = ChatGenerationConfig(temperature=1.0)


def _passage() -> SourcePassage:
    return SourcePassage(
        chunk_id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        chunk_index=3,
        content="AWS operating income increased.",
        section="Item 7. MD&A",
        ticker="AMZN",
        company_name="Amazon.com, Inc.",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001018724-24-000123",
        filing_date=date(2024, 2, 2),
        source_url="https://example.com/amzn-10k",
        score=0.42,
    )


@dataclass
class StubRetriever:
    passages: list[SourcePassage] = field(default_factory=list)

    def search_filings(self, query: str, *, limit: int = 10) -> RetrievalResult:
        return RetrievalResult(query=query, passages=self.passages)

    def search_filings_batch(
        self,
        queries: list[str],
        *,
        limit_per_query: int = 5,
    ) -> list[SourcePassage]:
        return self.passages[: limit_per_query * len(queries)]

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
class RecordingValidator:
    should_fail: bool = False
    received_passages: list[SourcePassage] | None = None
    validate_called: bool = False

    def validate(
        self,
        answer: GroundedAnswer,
        retrieved_passages: list[SourcePassage],
    ) -> None:
        self.validate_called = True
        self.received_passages = retrieved_passages
        if self.should_fail:
            raise GroundingError("ungrounded answer")


async def _collect(response) -> str:
    parts = [chunk async for chunk in response.body_iterator]
    return "".join(part.decode() if isinstance(part, bytes) else part for part in parts)


def _assembled_text(sse_body: str) -> str:
    text = ""
    for line in sse_body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[len("data: ") :])
        if event.get("type") == "text-delta":
            text += event["delta"]
    return text


def _appended_message() -> ChatMessageRecord:
    return ChatMessageRecord(
        id=MESSAGE_ID,
        thread_id=THREAD_ID,
        role="assistant",
        content="AWS operating income rose [1].",
        created_at=NOW,
    )


def _activity_kinds(sse_body: str) -> list[str]:
    kinds: list[str] = []
    for line in sse_body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[len("data: ") :])
        if event.get("type") == "data-activity" and event["data"].get("phase") == "start":
            kinds.append(event["data"]["kind"])
    return kinds


def _event_types(sse_body: str) -> list[str]:
    types: list[str] = []
    for line in sse_body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[len("data: ") :])
        event_type = event.get("type")
        if isinstance(event_type, str):
            types.append(event_type)
    return types


def _grounded_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer="AWS operating income rose [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=CHUNK_ID,
                excerpt="AWS operating income increased.",
            )
        ],
        cited_passages=[_passage()],
    )


async def _fake_run_agent(
    user_text: str,
    *,
    user_id: UUID,
    thread_id: UUID,
    chat_model: ResolvedChatModel,
    generation: ChatGenerationConfig,
    grounding_validator,  # unused in stub
    activity=None,
    retriever=None,
) -> tuple[GroundedAnswer, list[SourcePassage], EvidenceRegistry, TurnUsage]:
    if activity is not None:
        activity.start_thinking(f"Thinking with {chat_model.model}...")
    if retriever is not None:
        retriever.search_filings(user_text)
    if activity is not None:
        activity.end_thinking()
    evidence = EvidenceRegistry()
    usage = TurnUsage()
    return _grounded_answer(), [_passage()], evidence, usage


@pytest.mark.anyio
async def test_run_chat_turn_emits_progress_before_answer() -> None:
    retriever = StubRetriever(passages=[_passage()])
    validator = RecordingValidator(should_fail=False)
    client = object()

    append = AsyncMock(return_value=_appended_message())
    attach = AsyncMock(return_value=[])
    update_message_data = AsyncMock(return_value=_appended_message())

    with patch("app.chat.orchestrator.chat_store.append_message", new=append), patch(
        "app.chat.orchestrator.chat_store.attach_citations", new=attach
    ), patch(
        "app.chat.orchestrator.chat_store.update_message_data",
        new=update_message_data,
    ), patch(
        "app.chat.orchestrator.chat_store.title_thread_from_first_message",
        new=AsyncMock(),
    ), patch("app.chat.orchestrator._run_agent", side_effect=_fake_run_agent):
        response = await run_chat_turn(
            client,
            user_id=USER_ID,
            thread_id=THREAD_ID,
            user_text="How did AWS operating income change?",
            user_message_data=None,
            retriever=retriever,
            grounding_validator=validator,
            chat_model=TEST_CHAT_MODEL,
            generation=TEST_GENERATION,
        )
        body = await _collect(response)

    assert _assembled_text(body) == "AWS operating income rose [1]."
    assert "thinking" in _activity_kinds(body)
    assert "validate" in _activity_kinds(body)
    assert "save" in _activity_kinds(body)
    assert _event_types(body).index("start") < _event_types(body).index("text-delta")
    assert "data: [DONE]" in body
    assert validator.validate_called is True
    assert append.await_count == 2
    roles = [call.kwargs["role"] for call in append.await_args_list]
    assert roles == ["user", "assistant"]
    attach.assert_awaited_once()
    update_message_data.assert_awaited_once()
    citations = attach.await_args.kwargs["citations"]
    assert len(citations) == 1
    assert citations[0].chunk_id == CHUNK_ID
    assert citations[0].citation_index == 1


@pytest.mark.anyio
async def test_run_chat_turn_titles_thread_from_first_message() -> None:
    retriever = StubRetriever(passages=[_passage()])
    validator = RecordingValidator(should_fail=False)
    client = object()

    append = AsyncMock(return_value=_appended_message())
    title_thread = AsyncMock()

    with patch("app.chat.orchestrator.chat_store.append_message", new=append), patch(
        "app.chat.orchestrator.chat_store.title_thread_from_first_message",
        new=title_thread,
    ), patch(
        "app.chat.orchestrator._run_agent",
        side_effect=_fake_run_agent,
    ):
        response = await run_chat_turn(
            client,
            user_id=USER_ID,
            thread_id=THREAD_ID,
            user_text="Compare NVDA and AMD revenue growth trends",
            user_message_data=None,
            retriever=retriever,
            grounding_validator=validator,
            chat_model=TEST_CHAT_MODEL,
            generation=TEST_GENERATION,
        )

    title_thread.assert_awaited_once()
    assert title_thread.await_args.kwargs["user_text"] == (
        "Compare NVDA and AMD revenue growth trends"
    )
    del response


async def _fake_run_agent_ungrounded(
    user_text: str,
    *,
    user_id: UUID,
    thread_id: UUID,
    chat_model: ResolvedChatModel,
    generation: ChatGenerationConfig,
    grounding_validator,  # unused in stub
    activity=None,
    retriever=None,
) -> tuple[GroundedAnswer, list[SourcePassage], EvidenceRegistry, TurnUsage]:
    if activity is not None:
        activity.start_thinking(f"Thinking with {chat_model.model}...")
    if retriever is not None:
        retriever.search_filings(user_text)
    if activity is not None:
        activity.end_thinking()
    evidence = EvidenceRegistry()
    usage = TurnUsage()
    return (
        GroundedAnswer(answer="AWS operating income rose sharply without evidence."),
        [_passage()],
        evidence,
        usage,
    )


@pytest.mark.anyio
async def test_run_chat_turn_refuses_on_grounding_failure() -> None:
    retriever = StubRetriever(passages=[_passage()])
    validator = RecordingValidator(should_fail=True)
    client = object()

    append = AsyncMock(return_value=_appended_message())
    attach = AsyncMock(return_value=[])
    update_message_data = AsyncMock(return_value=_appended_message())

    async def fake_run_correction(**kwargs) -> GroundedDraft:  # type: ignore[override]
        # Return a corrective draft with the same answer text; validator will still fail.
        return GroundedDraft(answer=_grounded_answer().answer, citations=[])

    with patch("app.chat.orchestrator.chat_store.append_message", new=append), patch(
        "app.chat.orchestrator.chat_store.attach_citations", new=attach
    ), patch(
        "app.chat.orchestrator.chat_store.update_message_data",
        new=update_message_data,
    ), patch(
        "app.chat.orchestrator.chat_store.title_thread_from_first_message",
        new=AsyncMock(),
    ), patch(
        "app.chat.orchestrator._run_agent",
        side_effect=_fake_run_agent,
    ), patch(
        "app.chat.orchestrator.run_citation_correction",
        side_effect=fake_run_correction,
    ):
        response = await run_chat_turn(
            client,
            user_id=USER_ID,
            thread_id=THREAD_ID,
            user_text="How did AWS operating income change?",
            user_message_data=None,
            retriever=retriever,
            grounding_validator=validator,
            chat_model=TEST_CHAT_MODEL,
            generation=TEST_GENERATION,
        )
        body = await _collect(response)

    assert _assembled_text(body) == REFUSAL_MESSAGE
    assert "AWS operating income rose" not in body
    assert "[1]" not in body
    assert "data-citation" not in _event_types(body)
    assert "save" not in _activity_kinds(body)
    contents = [call.kwargs["content"] for call in append.await_args_list]
    assert contents == ["How did AWS operating income change?", REFUSAL_MESSAGE]
    attach.assert_not_awaited()
    update_message_data.assert_not_awaited()
    assert validator.validate_called is True


@pytest.mark.anyio
async def test_run_chat_turn_refuses_ungrounded_model_answer() -> None:
    retriever = StubRetriever(passages=[_passage()])
    client = object()

    append = AsyncMock(return_value=_appended_message())
    attach = AsyncMock(return_value=[])
    update_message_data = AsyncMock(return_value=_appended_message())

    with patch("app.chat.orchestrator.chat_store.append_message", new=append), patch(
        "app.chat.orchestrator.chat_store.attach_citations", new=attach
    ), patch(
        "app.chat.orchestrator.chat_store.update_message_data",
        new=update_message_data,
    ), patch(
        "app.chat.orchestrator.chat_store.title_thread_from_first_message",
        new=AsyncMock(),
    ), patch(
        "app.chat.orchestrator._run_agent",
        side_effect=_fake_run_agent_ungrounded,
    ):
        response = await run_chat_turn(
            client,
            user_id=USER_ID,
            thread_id=THREAD_ID,
            user_text="How did AWS operating income change?",
            user_message_data=None,
            retriever=retriever,
            grounding_validator=grounding_validator,
            chat_model=TEST_CHAT_MODEL,
            generation=TEST_GENERATION,
        )
        body = await _collect(response)

    assert _assembled_text(body) == REFUSAL_MESSAGE
    assert "without evidence" not in body
    assert "data-citation" not in _event_types(body)
    attach.assert_not_awaited()
    update_message_data.assert_not_awaited()


def _apple_mix_bad_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer=(
            "Apple's revenue mix shifted notably from 2021 to 2025, with Services growing "
            "its share of total net sales from 18.7% to 24%. iPhone share declined [1]."
        ),
        citations=[
            Citation(
                citation_index=1,
                chunk_id=CHUNK_ID,
                excerpt="iPhone share declined.",
            )
        ],
        cited_passages=[_passage()],
    )


def _apple_mix_fixed_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer=(
            "Apple's revenue mix shifted notably from 2021 to 2025, with Services growing "
            "its share of total net sales from 18.7% to 24% [1]. iPhone share declined [1]."
        ),
        citations=[
            Citation(
                citation_index=1,
                chunk_id=CHUNK_ID,
                excerpt="Services share grew from 18.7% to 24%.",
            )
        ],
        cited_passages=[_passage()],
    )


@pytest.mark.anyio
async def test_grounding_failure_uses_one_correction_without_extra_retrieval() -> None:
    """On grounding failure, run exactly one correction without re-running the agent."""
    retriever = StubRetriever(passages=[_passage()])
    client = object()

    append = AsyncMock(return_value=_appended_message())
    attach = AsyncMock(return_value=[])
    update_message_data = AsyncMock(return_value=_appended_message())

    initial_answer = _apple_mix_bad_answer()
    corrected = _apple_mix_fixed_answer()

    async def fake_run_agent(
        user_text: str,
        *,
        user_id: UUID,
        thread_id: UUID,
        chat_model: ResolvedChatModel,
        generation: ChatGenerationConfig,
        grounding_validator,
        activity=None,
        retriever=None,
    ) -> tuple[GroundedAnswer, list[SourcePassage], EvidenceRegistry, TurnUsage]:
        evidence = EvidenceRegistry()
        usage = TurnUsage()
        return initial_answer, [_passage()], evidence, usage

    finalize_calls = 0

    def fake_finalize_answer(
        answer: GroundedAnswer,
        retrieved_passages: list[SourcePassage],
        validator,
    ) -> GroundedAnswer:
        nonlocal finalize_calls
        finalize_calls += 1
        if finalize_calls == 1:
            raise GroundingError("Uncited segment")
        return answer

    correction_calls: list[tuple[str, str]] = []

    async def fake_run_correction(**kwargs) -> GroundedDraft:  # type: ignore[override]
        correction_calls.append((kwargs["failed_answer"], str(kwargs["grounding_error"])))
        return GroundedDraft(answer=corrected.answer, citations=[])

    with patch("app.chat.orchestrator.chat_store.append_message", new=append), patch(
        "app.chat.orchestrator.chat_store.attach_citations",
        new=attach,
    ), patch(
        "app.chat.orchestrator.chat_store.update_message_data",
        new=update_message_data,
    ), patch(
        "app.chat.orchestrator.chat_store.title_thread_from_first_message",
        new=AsyncMock(),
    ), patch(
        "app.chat.orchestrator._run_agent",
        side_effect=fake_run_agent,
    ) as run_agent_mock, patch(
        "app.chat.orchestrator._finalize_grounded_answer",
        side_effect=fake_finalize_answer,
    ), patch(
        "app.chat.orchestrator.run_citation_correction",
        side_effect=fake_run_correction,
    ) as correction_mock:
        response = await run_chat_turn(
            client,
            user_id=USER_ID,
            thread_id=THREAD_ID,
            user_text="How did Apple's revenue mix change from 2021 to 2025?",
            user_message_data=None,
            retriever=retriever,
            grounding_validator=grounding_validator,
            chat_model=TEST_CHAT_MODEL,
            generation=TEST_GENERATION,
        )
        body = await _collect(response)

    assert _assembled_text(body) == corrected.answer
    assert finalize_calls == 2  # initial + correction
    assert len(correction_calls) == 1
    assert run_agent_mock.await_count == 1
    assert (
        correction_mock.await_args.kwargs["max_tokens"]
        == DEFAULT_TURN_BUDGET.correction_output_tokens
    )
    attach.assert_awaited_once()
    update_message_data.assert_awaited_once()


@pytest.mark.anyio
async def test_grounding_correction_failure_refuses() -> None:
    """If correction still fails grounding, refuse with no second agent run."""
    retriever = StubRetriever(passages=[_passage()])
    client = object()

    append = AsyncMock(return_value=_appended_message())
    attach = AsyncMock(return_value=[])
    update_message_data = AsyncMock(return_value=_appended_message())

    initial_answer = _apple_mix_bad_answer()

    async def fake_run_agent(
        user_text: str,
        *,
        user_id: UUID,
        thread_id: UUID,
        chat_model: ResolvedChatModel,
        generation: ChatGenerationConfig,
        grounding_validator,
        activity=None,
        retriever=None,
    ) -> tuple[GroundedAnswer, list[SourcePassage], EvidenceRegistry, TurnUsage]:
        evidence = EvidenceRegistry()
        usage = TurnUsage()
        return initial_answer, [_passage()], evidence, usage

    finalize_calls = 0

    def always_fail_finalize(
        answer: GroundedAnswer,
        retrieved_passages: list[SourcePassage],
        validator,
    ) -> GroundedAnswer:
        nonlocal finalize_calls
        finalize_calls += 1
        raise GroundingError("still ungrounded")

    async def fake_run_correction(**kwargs) -> GroundedDraft:  # type: ignore[override]
        return GroundedDraft(answer=initial_answer.answer, citations=[])

    with patch("app.chat.orchestrator.chat_store.append_message", new=append), patch(
        "app.chat.orchestrator.chat_store.attach_citations",
        new=attach,
    ), patch(
        "app.chat.orchestrator.chat_store.update_message_data",
        new=update_message_data,
    ), patch(
        "app.chat.orchestrator.chat_store.title_thread_from_first_message",
        new=AsyncMock(),
    ), patch(
        "app.chat.orchestrator._run_agent",
        side_effect=fake_run_agent,
    ) as run_agent_mock, patch(
        "app.chat.orchestrator._finalize_grounded_answer",
        side_effect=always_fail_finalize,
    ), patch(
        "app.chat.orchestrator.run_citation_correction",
        side_effect=fake_run_correction,
    ) as correction_mock:
        response = await run_chat_turn(
            client,
            user_id=USER_ID,
            thread_id=THREAD_ID,
            user_text="How did Apple's revenue mix change from 2021 to 2025?",
            user_message_data=None,
            retriever=retriever,
            grounding_validator=grounding_validator,
            chat_model=TEST_CHAT_MODEL,
            generation=TEST_GENERATION,
        )
        body = await _collect(response)

    assert _assembled_text(body) == REFUSAL_MESSAGE
    assert finalize_calls == 2  # initial + correction
    assert run_agent_mock.await_count == 1
    assert correction_mock.await_count == 1
    attach.assert_not_awaited()
    update_message_data.assert_not_awaited()


@pytest.mark.anyio
async def test_run_chat_turn_does_not_retry_without_citations() -> None:
    """Do not attempt correction when the model answer has no citations."""
    retriever = StubRetriever(passages=[_passage()])
    client = object()

    async def fake_run_agent_uncited(
        user_text: str,
        *,
        user_id: UUID,
        thread_id: UUID,
        chat_model: ResolvedChatModel,
        generation: ChatGenerationConfig,
        grounding_validator,
        activity=None,
        retriever=None,
    ) -> tuple[GroundedAnswer, list[SourcePassage], EvidenceRegistry, TurnUsage]:
        evidence = EvidenceRegistry()
        usage = TurnUsage()
        return (
            GroundedAnswer(answer="AWS operating income rose 42%.", citations=[]),
            [_passage()],
            evidence,
            usage,
        )

    def fail_finalize(
        answer: GroundedAnswer,
        retrieved_passages: list[SourcePassage],
        validator,
    ) -> GroundedAnswer:
        raise GroundingError("uncited")

    append = AsyncMock(return_value=_appended_message())
    attach = AsyncMock(return_value=[])

    with patch("app.chat.orchestrator.chat_store.append_message", new=append), patch(
        "app.chat.orchestrator.chat_store.attach_citations",
        new=attach,
    ), patch(
        "app.chat.orchestrator.chat_store.update_message_data",
        new=AsyncMock(),
    ), patch(
        "app.chat.orchestrator.chat_store.title_thread_from_first_message",
        new=AsyncMock(),
    ), patch(
        "app.chat.orchestrator._run_agent",
        side_effect=fake_run_agent_uncited,
    ), patch(
        "app.chat.orchestrator._finalize_grounded_answer",
        side_effect=fail_finalize,
    ), patch(
        "app.chat.orchestrator.run_citation_correction",
        new_callable=AsyncMock,
    ) as correction_mock:
        response = await run_chat_turn(
            client,
            user_id=USER_ID,
            thread_id=THREAD_ID,
            user_text="How did AWS operating income change?",
            user_message_data=None,
            retriever=retriever,
            grounding_validator=grounding_validator,
            chat_model=TEST_CHAT_MODEL,
            generation=TEST_GENERATION,
        )
        body = await _collect(response)

    assert _assembled_text(body) == REFUSAL_MESSAGE
    correction_mock.assert_not_awaited()
    attach.assert_not_awaited()


@pytest.mark.anyio
async def test_run_chat_turn_streams_model_unavailable_message() -> None:
    retriever = StubRetriever(passages=[_passage()])
    validator = RecordingValidator(should_fail=False)
    client = object()

    append = AsyncMock(return_value=_appended_message())
    quota_message = model_unavailable_message(
        ModelHTTPError(
            status_code=429,
            model_name="gemini-2.0-flash",
            body={"error": {"code": 429}},
        ),
        provider="google",
    )

    with patch("app.chat.orchestrator.chat_store.append_message", new=append), patch(
        "app.chat.orchestrator.chat_store.title_thread_from_first_message",
        new=AsyncMock(),
    ), patch(
        "app.chat.orchestrator._run_agent",
        new=AsyncMock(
            side_effect=ModelHTTPError(
                status_code=429,
                model_name="gemini-2.0-flash",
                body={"error": {"code": 429}},
            )
        ),
    ):
        response = await run_chat_turn(
            client,
            user_id=USER_ID,
            thread_id=THREAD_ID,
            user_text="How did AWS operating income change?",
            user_message_data=None,
            retriever=retriever,
            grounding_validator=validator,
            chat_model=TEST_CHAT_MODEL,
            generation=TEST_GENERATION,
        )
        body = await _collect(response)

    assert _assembled_text(body) == quota_message
    contents = [call.kwargs["content"] for call in append.await_args_list]
    assert contents == ["How did AWS operating income change?", quota_message]


ROUTER_MODEL = ResolvedChatModel(provider="google", model="gemini-2.0-flash-lite")
SYNTHESIS_MODEL = ResolvedChatModel(provider="google", model="gemini-3.5-flash-lite")


def _coverage() -> CorpusCoverage:
    return CorpusCoverage(ticker_years={"AMZN": frozenset({2024})})


def _extractive_plan() -> QueryPlan:
    return QueryPlan(
        route="extractive",
        tickers=["AMZN"],
        fiscal_years=[2024],
        topics=["AWS operating income"],
        primary_queries=["AMZN AWS operating income 2024"],
        reserve_queries=[],
        requires_synthesis=False,
    )


def _extractive_result_with_draft() -> FactExtraction:
    return FactExtraction(
        facts=[
            ExtractedFact(
                status="supported",
                ticker="AMZN",
                fiscal_year=2024,
                topic="AWS operating income",
                value=None,
                unit=None,
                finding="AWS operating income increased.",
                evidence_alias="E1",
            )
        ],
        missing_scope=[],
        conflicts=[],
        draft=GroundedDraft(
            answer="AWS operating income increased [1].",
            citations=[
                DraftCitation(
                    citation_index=1,
                    evidence_alias="E1",
                    excerpt="AWS operating income increased.",
                )
            ],
        ),
    )


@pytest.mark.anyio
async def test_extractive_route_does_not_call_synthesis() -> None:
    router = AsyncMock(return_value=_extractive_plan())
    extractor = AsyncMock(return_value=_extractive_result_with_draft())
    synthesis = AsyncMock()

    with patch(
        "app.chat.orchestrator.resolve_router_model",
        return_value=ROUTER_MODEL,
    ), patch(
        "app.chat.orchestrator.build_document_agent_model",
        return_value=object(),
    ), patch(
        "app.chat.orchestrator.run_query_router",
        new=router,
    ), patch(
        "app.chat.orchestrator.run_fact_extractor",
        new=extractor,
    ), patch(
        "app.chat.orchestrator.run_synthesis",
        new=synthesis,
    ):
        answer, _, _, usage = await _run_routed_turn(
            "Compare AWS operating income",
            user_id=USER_ID,
            thread_id=THREAD_ID,
            chat_model=SYNTHESIS_MODEL,
            generation=ChatGenerationConfig(),
            grounding_validator=grounding_validator,
            retriever=StubRetriever(passages=[_passage()]),
            coverage=_coverage(),
            activity=None,
        )

    synthesis.assert_not_awaited()
    assert answer.answer
    assert usage.route == "extractive"


def _synthesis_plan(*, tickers: list[str] | None = None) -> QueryPlan:
    return QueryPlan(
        route="synthesis",
        tickers=tickers or ["AMZN"],
        fiscal_years=[2024],
        topics=["AWS operating income"],
        primary_queries=["AMZN AWS operating income 2024"],
        reserve_queries=[],
        requires_synthesis=True,
    )


def _boundary_plan() -> QueryPlan:
    return QueryPlan(
        route="boundary",
        tickers=["AMZN"],
        fiscal_years=[2024],
        topics=["AWS operating income"],
        primary_queries=["AMZN AWS operating income 2024"],
        reserve_queries=[],
        requires_synthesis=False,
    )


def _synthesis_extraction() -> FactExtraction:
    return FactExtraction(
        facts=[
            ExtractedFact(
                status="supported",
                ticker="AMZN",
                fiscal_year=2024,
                topic="AWS operating income",
                value=None,
                unit=None,
                finding="AWS operating income increased.",
                evidence_alias="E1",
            )
        ],
        missing_scope=[],
        conflicts=[],
        draft=None,
    )


def _draft() -> GroundedDraft:
    return GroundedDraft(
        answer="Answer [1].",
        citations=[
            DraftCitation(
                citation_index=1,
                evidence_alias="E1",
                excerpt="AWS operating income increased.",
            )
        ],
    )


@pytest.mark.anyio
async def test_synthesis_route_calls_selected_chat_model() -> None:
    router = AsyncMock(return_value=_synthesis_plan())
    extractor = AsyncMock(return_value=_synthesis_extraction())
    synthesis = AsyncMock(return_value=_draft())
    router_llm = object()
    synthesis_llm = object()

    def fake_build_model(provider: str, model: str) -> object:
        if model == ROUTER_MODEL.model:
            return router_llm
        if model == SYNTHESIS_MODEL.model:
            return synthesis_llm
        raise AssertionError(model)

    with patch(
        "app.chat.orchestrator.resolve_router_model",
        return_value=ROUTER_MODEL,
    ), patch(
        "app.chat.orchestrator.build_document_agent_model",
        side_effect=fake_build_model,
    ), patch(
        "app.chat.orchestrator.run_query_router",
        new=router,
    ), patch(
        "app.chat.orchestrator.run_fact_extractor",
        new=extractor,
    ), patch(
        "app.chat.orchestrator.run_synthesis",
        new=synthesis,
    ):
        answer, _, evidence, usage = await _run_routed_turn(
            "How did AWS operating income change?",
            user_id=USER_ID,
            thread_id=THREAD_ID,
            chat_model=SYNTHESIS_MODEL,
            generation=ChatGenerationConfig(),
            grounding_validator=grounding_validator,
            retriever=StubRetriever(passages=[_passage()]),
            coverage=_coverage(),
            activity=None,
        )

    assert answer.answer
    assert evidence.max_passages == 8
    assert usage.route == "synthesis"
    assert usage.budget_profile == "standard"
    assert synthesis.await_args.args[4] is synthesis_llm
    assert extractor.await_args.args[3] is router_llm


@pytest.mark.anyio
async def test_boundary_route_can_return_grounded_limitation() -> None:
    router = AsyncMock(return_value=_boundary_plan())
    extractor = AsyncMock(return_value=_synthesis_extraction())
    synthesis = AsyncMock(return_value=_draft())

    with patch(
        "app.chat.orchestrator.resolve_router_model",
        return_value=ROUTER_MODEL,
    ), patch(
        "app.chat.orchestrator.build_document_agent_model",
        return_value=object(),
    ), patch(
        "app.chat.orchestrator.run_query_router",
        new=router,
    ), patch(
        "app.chat.orchestrator.run_fact_extractor",
        new=extractor,
    ), patch(
        "app.chat.orchestrator.run_synthesis",
        new=synthesis,
    ):
        answer, _, _, usage = await _run_routed_turn(
            "Prove insider intent beyond filings",
            user_id=USER_ID,
            thread_id=THREAD_ID,
            chat_model=SYNTHESIS_MODEL,
            generation=ChatGenerationConfig(),
            grounding_validator=grounding_validator,
            retriever=StubRetriever(passages=[_passage()]),
            coverage=_coverage(),
            activity=None,
        )

    assert answer.answer
    assert usage.route == "boundary"


@pytest.mark.anyio
async def test_router_model_failure_uses_direct_fallback_without_extractor() -> None:
    router = AsyncMock(
        side_effect=ModelHTTPError(
            status_code=429,
            model_name=ROUTER_MODEL.model,
            body={"error": {"code": 429}},
        )
    )
    extractor = AsyncMock()
    fallback = AsyncMock(return_value=_draft())

    with patch(
        "app.chat.orchestrator.resolve_router_model",
        return_value=ROUTER_MODEL,
    ), patch(
        "app.chat.orchestrator.build_document_agent_model",
        return_value=object(),
    ), patch(
        "app.chat.orchestrator.run_query_router",
        new=router,
    ), patch(
        "app.chat.orchestrator.run_fact_extractor",
        new=extractor,
    ), patch(
        "app.chat.orchestrator.run_direct_fallback",
        new=fallback,
    ):
        _, _, _, usage = await _run_routed_turn(
            "How did AWS operating income change?",
            user_id=USER_ID,
            thread_id=THREAD_ID,
            chat_model=SYNTHESIS_MODEL,
            generation=ChatGenerationConfig(),
            grounding_validator=grounding_validator,
            retriever=StubRetriever(passages=[_passage()]),
            coverage=_coverage(),
            activity=None,
        )

    extractor.assert_not_awaited()
    fallback.assert_awaited_once()
    assert usage.route == "synthesis"
    assert usage.budget_profile == "standard"


@pytest.mark.anyio
async def test_router_model_unavailable_uses_direct_fallback_without_router_call() -> None:
    router = AsyncMock()
    extractor = AsyncMock()
    fallback = AsyncMock(return_value=_draft())

    with patch(
        "app.chat.orchestrator.resolve_router_model",
        return_value=None,
    ), patch(
        "app.chat.orchestrator.build_document_agent_model",
        return_value=object(),
    ), patch(
        "app.chat.orchestrator.run_query_router",
        new=router,
    ), patch(
        "app.chat.orchestrator.run_fact_extractor",
        new=extractor,
    ), patch(
        "app.chat.orchestrator.run_direct_fallback",
        new=fallback,
    ):
        _, _, _, usage = await _run_routed_turn(
            "How did AWS operating income change?",
            user_id=USER_ID,
            thread_id=THREAD_ID,
            chat_model=SYNTHESIS_MODEL,
            generation=ChatGenerationConfig(),
            grounding_validator=grounding_validator,
            retriever=StubRetriever(passages=[_passage()]),
            coverage=_coverage(),
            activity=None,
        )

    router.assert_not_awaited()
    extractor.assert_not_awaited()
    fallback.assert_awaited_once()
    assert usage.route == "synthesis"


@pytest.mark.anyio
async def test_broad_multi_company_plan_uses_15_passage_budget() -> None:
    router = AsyncMock(return_value=_synthesis_plan(tickers=["AMZN", "MSFT"]))
    extractor = AsyncMock(return_value=_synthesis_extraction())
    synthesis = AsyncMock(return_value=_draft())

    passages = [_passage() for _ in range(20)]
    with patch(
        "app.chat.orchestrator.resolve_router_model",
        return_value=ROUTER_MODEL,
    ), patch(
        "app.chat.orchestrator.build_document_agent_model",
        return_value=object(),
    ), patch(
        "app.chat.orchestrator.run_query_router",
        new=router,
    ), patch(
        "app.chat.orchestrator.run_fact_extractor",
        new=extractor,
    ), patch(
        "app.chat.orchestrator.run_synthesis",
        new=synthesis,
    ):
        _, _, evidence, usage = await _run_routed_turn(
            "Compare AWS and Microsoft cloud margins",
            user_id=USER_ID,
            thread_id=THREAD_ID,
            chat_model=SYNTHESIS_MODEL,
            generation=ChatGenerationConfig(),
            grounding_validator=grounding_validator,
            retriever=StubRetriever(passages=passages),
            coverage=_coverage(),
            activity=None,
        )

    assert evidence.max_passages == 15
    assert usage.budget_profile == "broad"


@pytest.mark.anyio
async def test_retrieval_runs_via_asyncio_to_thread() -> None:
    router = AsyncMock(return_value=_synthesis_plan())
    extractor = AsyncMock(return_value=_synthesis_extraction())
    synthesis = AsyncMock(return_value=_draft())

    async def fake_to_thread(fn, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs == {}
        return fn(*args)

    with patch(
        "app.chat.orchestrator.asyncio.to_thread",
        side_effect=fake_to_thread,
    ) as to_thread_mock, patch(
        "app.chat.orchestrator.resolve_router_model",
        return_value=ROUTER_MODEL,
    ), patch(
        "app.chat.orchestrator.build_document_agent_model",
        return_value=object(),
    ), patch(
        "app.chat.orchestrator.run_query_router",
        new=router,
    ), patch(
        "app.chat.orchestrator.run_fact_extractor",
        new=extractor,
    ), patch(
        "app.chat.orchestrator.run_synthesis",
        new=synthesis,
    ):
        await _run_routed_turn(
            "How did AWS operating income change?",
            user_id=USER_ID,
            thread_id=THREAD_ID,
            chat_model=SYNTHESIS_MODEL,
            generation=ChatGenerationConfig(),
            grounding_validator=grounding_validator,
            retriever=StubRetriever(passages=[_passage()]),
            coverage=_coverage(),
            activity=None,
        )

    assert to_thread_mock.await_count == 1


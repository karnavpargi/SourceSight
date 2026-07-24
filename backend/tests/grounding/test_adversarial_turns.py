"""Phase 7 adversarial and happy-path grounding turn tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest

from app.assistant.evidence import EvidenceRegistry
from app.assistant.outputs import Citation, GroundedAnswer
from app.chat.generation import ChatGenerationConfig
from app.chat.models_catalog import ResolvedChatModel
from app.chat.orchestrator import REFUSAL_MESSAGE, run_chat_turn
from app.chat.usage import TurnUsage
from app.database.chats import ChatMessageRecord
from app.grounding.validator import grounding_validator
from app.retrieval.types import RetrievalResult, SourcePassage

USER_ID = UUID("550e8400-e29b-41d4-a716-446655440000")
THREAD_ID = UUID("770e8400-e29b-41d4-a716-446655440002")
MESSAGE_ID = UUID("880e8400-e29b-41d4-a716-446655440003")
CHUNK_ID = UUID("11111111-1111-1111-1111-111111111111")
DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
TEST_CHAT_MODEL = ResolvedChatModel(provider="google", model="gemini-2.0-flash")
TEST_GENERATION = ChatGenerationConfig(temperature=1.0)

UNKNOWN_COMPANY_PROMPT = "What was Tesla's revenue growth in fiscal 2024?"
STOCK_RECOMMENDATION_PROMPT = "Should I buy NVDA stock right now?"
WELL_GROUNDED_PROMPT = "How did AWS operating income change in Amazon's latest 10-K?"


@dataclass
class StubRetriever:
    passages: list[SourcePassage] = field(default_factory=list)

    def search_filings(self, query: str, *, limit: int = 10) -> RetrievalResult:
        return RetrievalResult(query=query, passages=self.passages)

    def read_chunk(self, chunk_id: UUID) -> SourcePassage:
        raise NotImplementedError(chunk_id)

    def read_surrounding_chunks(
        self,
        chunk_id: UUID,
        *,
        window: int = 1,
    ) -> list[SourcePassage]:
        raise NotImplementedError((chunk_id, window))


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


def _appended_message() -> ChatMessageRecord:
    return ChatMessageRecord(
        id=MESSAGE_ID,
        thread_id=THREAD_ID,
        role="assistant",
        content="AWS operating income rose [1].",
        created_at=NOW,
    )


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


async def _run_turn(
    *,
    user_text: str,
    agent_side_effect,
    passages: list[SourcePassage] | None = None,
) -> tuple[str, AsyncMock, AsyncMock, AsyncMock]:
    retriever = StubRetriever(passages=passages or [])
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
    ), patch("app.chat.orchestrator._run_agent", side_effect=agent_side_effect):
        response = await run_chat_turn(
            client,
            user_id=USER_ID,
            thread_id=THREAD_ID,
            user_text=user_text,
            user_message_data=None,
            retriever=retriever,
            grounding_validator=grounding_validator,
            chat_model=TEST_CHAT_MODEL,
            generation=TEST_GENERATION,
        )
        body = await _collect(response)

    return body, append, attach, update_message_data


def _fake_run_agent_factory(answer: GroundedAnswer, passages: list[SourcePassage]):
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
        if retriever is not None:
            retriever.search_filings(user_text)
        evidence = EvidenceRegistry()
        usage = TurnUsage()
        return answer, passages, evidence, usage

    return _fake_run_agent


@pytest.mark.anyio
async def test_refuses_prompt_about_company_not_in_corpus() -> None:
    """Adversarial: Tesla is outside the indexed corpus; uncited claims are blocked."""
    hallucinated = GroundedAnswer(
        answer="Tesla reported 21% revenue growth in fiscal 2024."
    )
    body, append, attach, update_message_data = await _run_turn(
        user_text=UNKNOWN_COMPANY_PROMPT,
        agent_side_effect=_fake_run_agent_factory(hallucinated, []),
    )

    assert _assembled_text(body) == REFUSAL_MESSAGE
    assert "Tesla" not in body
    assert "data-citation" not in _event_types(body)
    attach.assert_not_awaited()
    update_message_data.assert_not_awaited()
    assistant_content = append.await_args_list[-1].kwargs["content"]
    assert assistant_content == REFUSAL_MESSAGE


@pytest.mark.anyio
async def test_refuses_stock_recommendation_prompt() -> None:
    """Adversarial: investment advice without citations must not reach the client."""
    stock_pick = GroundedAnswer(
        answer="NVDA is a strong buy with accelerating AI demand."
    )
    body, append, attach, update_message_data = await _run_turn(
        user_text=STOCK_RECOMMENDATION_PROMPT,
        agent_side_effect=_fake_run_agent_factory(stock_pick, [_passage()]),
    )

    assert _assembled_text(body) == REFUSAL_MESSAGE
    assert "strong buy" not in body
    assert "data-citation" not in _event_types(body)
    attach.assert_not_awaited()
    update_message_data.assert_not_awaited()
    assistant_content = append.await_args_list[-1].kwargs["content"]
    assert assistant_content == REFUSAL_MESSAGE


@pytest.mark.anyio
async def test_refuses_model_answer_without_citations() -> None:
    """Adversarial: a grounded-looking answer with zero citations is refused."""
    uncited = GroundedAnswer(
        answer="AWS operating income rose sharply year over year."
    )
    body, append, attach, update_message_data = await _run_turn(
        user_text=WELL_GROUNDED_PROMPT,
        agent_side_effect=_fake_run_agent_factory(uncited, [_passage()]),
    )

    assert _assembled_text(body) == REFUSAL_MESSAGE
    assert "rose sharply" not in body
    attach.assert_not_awaited()
    update_message_data.assert_not_awaited()


@pytest.mark.anyio
async def test_repairs_answer_with_markers_but_missing_citation_records() -> None:
    """Happy path: markers + cited_passages are repaired into a valid grounded answer."""
    passages = [_passage()]
    answer = GroundedAnswer(
        answer="AWS operating income rose [1].",
        cited_passages=passages,
    )
    body, append, attach, update_message_data = await _run_turn(
        user_text=WELL_GROUNDED_PROMPT,
        agent_side_effect=_fake_run_agent_factory(answer, passages),
        passages=passages,
    )

    assert _assembled_text(body) == "AWS operating income rose [1]."
    assert "data-citation" in _event_types(body)
    attach.assert_awaited_once()
    update_message_data.assert_awaited_once()


@pytest.mark.anyio
async def test_well_grounded_question_streams_normally() -> None:
    """Happy path: a cited answer streams with citation parts and is persisted."""
    body, append, attach, update_message_data = await _run_turn(
        user_text=WELL_GROUNDED_PROMPT,
        agent_side_effect=_fake_run_agent_factory(_grounded_answer(), [_passage()]),
        passages=[_passage()],
    )

    assert _assembled_text(body) == "AWS operating income rose [1]."
    assert "data-citation" in _event_types(body)
    assert REFUSAL_MESSAGE not in body
    attach.assert_awaited_once()
    update_message_data.assert_awaited_once()
    roles = [call.kwargs["role"] for call in append.await_args_list]
    assert roles == ["user", "assistant"]
    citations = attach.await_args.kwargs["citations"]
    assert len(citations) == 1
    assert citations[0].chunk_id == CHUNK_ID

"""Coordinates one chat turn end-to-end.

Turn lifecycle: stream start -> run agent with progress -> validate -> persist -> stream answer.

The retriever and grounding validator are injected so the orchestrator stays
testable without a live LLM or database. Production turns open their own DB
session inside the background agent task so streaming stays concurrent-safe.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import UUID

import structlog
from fastapi.responses import StreamingResponse
from supabase import AsyncClient

from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from app.assistant.agent import build_document_agent_model, document_agent
from app.assistant.deps import DocumentAgentDeps, DocumentRetriever, GroundingValidator
from app.assistant.evidence import EvidenceRegistry
from app.assistant.finalize import finalize_grounded_draft
from app.assistant.outputs import GroundedAnswer, GroundedDraft
from app.chat.activity_summary import group_activity_steps, merge_activity_log
from app.chat.agent_events import agent_event_stream_handler
from app.chat.generation import (
    ChatGenerationConfig,
    build_model_settings,
)
from app.chat.models_catalog import ResolvedChatModel
from app.chat.messages import TurnActivityData
from app.chat.persistence import assistant_answer_to_wire
from app.chat.turn_activity import TurnActivityEmitter
from app.chat.turn_budget import DEFAULT_TURN_BUDGET, TurnBudget
from app.chat.usage import TurnUsage
from app.chat.streaming import (
    format_activity_event,
    format_progress_event,
    format_ui_message_sse_event,
    stream_events,
    stream_grounded_answer_events,
    stream_ui_message_text,
)
from app.config import ChatProvider
from app.chat.thread_titles import DEFAULT_THREAD_TITLE, derive_thread_title
from app.database import chats as chat_store
from app.database.chats import AttachCitationInput
from app.database.session import session_scope
from app.grounding.validator import GroundingError
from app.grounding.repair import repair_grounded_answer
from app.retrieval.chunk_lookup import chunk_not_found_retry
from app.retrieval.document_retriever import SessionPerCallDocumentRetriever
from app.retrieval.retriever import _load_neighbor_passages
from app.retrieval.types import RetrievalResult, SourcePassage

REFUSAL_MESSAGE = "This corpus doesn't contain enough evidence to answer that."

_GROUNDING_RETRY_FEEDBACK = (
    "Rewrite the answer so every sentence containing $, %, or numeric amounts has inline "
    "[n] citation markers matching your citation records. Keep the same evidence and "
    "claims; only fix citation placement."
)

MODEL_UNAVAILABLE_MESSAGE = (
    "The language model is unavailable right now. "
    "Check your CHAT_PROVIDER API key, billing, and quota."
)

TURN_FAILED_MESSAGE = (
    "Something went wrong while processing your request. Please try again."
)

logger = structlog.get_logger(__name__)

__all__ = [
    "MODEL_UNAVAILABLE_MESSAGE",
    "REFUSAL_MESSAGE",
    "GroundingError",
    "model_unavailable_message",
    "run_chat_turn",
]


def model_unavailable_message(exc: BaseException, *, provider: ChatProvider) -> str:
    if isinstance(exc, ModelHTTPError):
        if exc.status_code == 429:
            if provider == "google":
                return (
                    "Google AI Studio quota exceeded. "
                    "Check usage at https://ai.dev/rate-limit or choose another provider."
                )
            if provider == "opencode":
                return (
                    "OpenCode Zen quota exceeded. "
                    "Check billing at https://opencode.ai/docs/zen/ or choose another provider."
                )
            return (
                "Local Ollama is overloaded or unavailable. "
                "Ensure `ollama serve` is running and the chat model is pulled."
            )
        if exc.status_code in {401, 403}:
            return (
                f"Invalid or unauthorized API key for provider {provider!r}. "
                "Check the key in backend/.env."
            )

    if isinstance(exc, UnexpectedModelBehavior) and provider == "local":
        return (
            "The local model failed to produce a valid response. "
            "Try another model or provider."
        )

    return MODEL_UNAVAILABLE_MESSAGE


def _token_usage_fields(run: object) -> dict[str, int | None]:
    usage = getattr(run, "usage", None)
    if callable(usage):
        usage = usage()
    if usage is None:
        return {"input_tokens": None, "output_tokens": None}

    input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "request_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "response_tokens", None)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _log_turn_complete(
    *,
    outcome: str,
    provider: ChatProvider,
    model: str,
    started_at: float,
    citation_count: int = 0,
    usage: TurnUsage | None = None,
    **extra: object,
) -> None:
    usage_fields: dict[str, int] = {}
    if usage is not None:
        usage_fields = usage.as_log_fields()

    logger.info(
        "chat.turn_complete",
        outcome=outcome,
        provider=provider,
        model=model,
        latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
        citation_count=citation_count,
        **usage_fields,
        **extra,
    )


@dataclass
class _RecordingRetriever:
    """Wraps a retriever to record every passage returned during the turn."""

    inner: DocumentRetriever
    activity: TurnActivityEmitter | None = None
    seen: dict[UUID, SourcePassage] = field(default_factory=dict)

    def search_filings(self, query: str, *, limit: int = 10) -> RetrievalResult:
        self._update("Searching indexed filings...", detail=query)
        result = self.inner.search_filings(query, limit=limit)
        self._record(result.passages)
        hit_count = len(result.passages)
        detail = f"{hit_count} passage{'s' if hit_count != 1 else ''} found"
        self._update("Analyzing retrieved passages...", detail=detail)
        return result

    def search_filings_batch(
        self,
        queries: list[str],
        *,
        limit_per_query: int = 5,
    ) -> list[SourcePassage]:
        joined = " | ".join(q for q in queries if q)
        self._update(
            "Searching indexed filings...",
            detail=joined or None,
        )
        passages = self.inner.search_filings_batch(
            queries,
            limit_per_query=limit_per_query,
        )
        self._record(passages)
        hit_count = len(passages)
        detail = f"{hit_count} passage{'s' if hit_count != 1 else ''} found"
        self._update("Analyzing retrieved passages...", detail=detail)
        return passages

    def read_chunk(self, chunk_id: UUID) -> SourcePassage:
        cached = self.seen.get(chunk_id)
        if cached is not None:
            return cached

        self._update("Reading filing passage...", detail=f"Chunk {chunk_id}")
        try:
            passage = self.inner.read_chunk(chunk_id)
        except ValueError as exc:
            raise chunk_not_found_retry(chunk_id) from exc
        self._record([passage])
        self._update("Analyzing retrieved passages...")
        return passage

    def read_surrounding_chunks(
        self,
        chunk_id: UUID,
        *,
        window: int = 1,
    ) -> list[SourcePassage]:
        self._update(
            "Reading surrounding context...",
            detail=f"Chunk {chunk_id} · window ±{window}",
        )
        try:
            passages = self.inner.read_surrounding_chunks(chunk_id, window=window)
        except ValueError as exc:
            cached = self.seen.get(chunk_id)
            if cached is None:
                raise chunk_not_found_retry(chunk_id) from exc
            passages = self._neighbors_for_cached_anchor(cached, window=window)
        self._record(passages)
        self._update("Analyzing retrieved passages...")
        return passages

    def _update(self, label: str, *, detail: str | None = None) -> None:
        if self.activity is not None:
            self.activity.update_active(label, detail=detail)

    def _neighbors_for_cached_anchor(
        self,
        anchor: SourcePassage,
        *,
        window: int,
    ) -> list[SourcePassage]:
        with session_scope() as session:
            return _load_neighbor_passages(
                session,
                [anchor],
                neighbor_window=window,
                existing_chunk_ids={anchor.chunk_id},
            )

    def _record(self, passages: list[SourcePassage]) -> None:
        for passage in passages:
            self.seen.setdefault(passage.chunk_id, passage)

    @property
    def retrieved_passages(self) -> list[SourcePassage]:
        return list(self.seen.values())


async def run_chat_turn(
    client: AsyncClient,
    *,
    user_id: UUID,
    thread_id: UUID,
    user_text: str,
    user_message_data: dict | None,
    grounding_validator: GroundingValidator,
    chat_model: ResolvedChatModel,
    generation: ChatGenerationConfig,
    retriever: DocumentRetriever | None = None,
) -> StreamingResponse:
    """Run one chat turn and return the assistant response as an AI SDK stream."""
    await chat_store.append_message(
        client,
        user_id=user_id,
        thread_id=thread_id,
        role="user",
        content=user_text,
        message_data=user_message_data,
    )
    await chat_store.title_thread_from_first_message(
        client,
        user_id,
        thread_id,
        user_text=user_text,
        default_title=DEFAULT_THREAD_TITLE,
        derive_title=derive_thread_title,
    )

    return stream_events(
        _stream_chat_turn(
            client,
            user_id=user_id,
            thread_id=thread_id,
            user_text=user_text,
            retriever=retriever,
            grounding_validator=grounding_validator,
            chat_model=chat_model,
            generation=generation,
        )
    )


async def _stream_chat_turn(
    client: AsyncClient,
    *,
    user_id: UUID,
    thread_id: UUID,
    user_text: str,
    retriever: DocumentRetriever | None,
    grounding_validator: GroundingValidator,
    chat_model: ResolvedChatModel,
    generation: ChatGenerationConfig,
) -> AsyncIterator[str]:
    turn_started = time.perf_counter()
    structlog.contextvars.bind_contextvars(
        user_id=str(user_id),
        thread_id=str(thread_id),
    )
    message_id = f"msg_{uuid.uuid4().hex}"
    activity = TurnActivityEmitter()
    activity_log: list[TurnActivityData] = []
    last_progress = "Analyzing your question..."
    usage: TurnUsage | None = None
    evidence: EvidenceRegistry | None = None

    def emit_activity_updates() -> list[str]:
        nonlocal last_progress
        events: list[str] = []
        for update in activity.drain():
            activity_log.append(update)
            events.append(format_activity_event(update))
            if update.label:
                last_progress = update.label
                events.append(format_progress_event(last_progress))
        return events

    yield format_ui_message_sse_event({"type": "start", "messageId": message_id})
    activity.start_thinking("Analyzing your question...")
    for event in emit_activity_updates():
        yield event
    await asyncio.sleep(0)

    agent_task = asyncio.create_task(
        _run_agent(
            user_text,
            user_id=user_id,
            thread_id=thread_id,
            chat_model=chat_model,
            generation=generation,
            grounding_validator=grounding_validator,
            activity=activity,
            retriever=retriever,
        )
    )

    try:
        while not agent_task.done():
            for event in emit_activity_updates():
                yield event
                await asyncio.sleep(0)
            await asyncio.sleep(0.05)

        for event in emit_activity_updates():
            yield event
            await asyncio.sleep(0)

        answer, retrieved_passages, evidence, usage = agent_task.result()
        for event in emit_activity_updates():
            yield event
            await asyncio.sleep(0)
    except (ModelHTTPError, UnexpectedModelBehavior) as exc:
        message = model_unavailable_message(exc, provider=chat_model.provider)
        logger.warning(
            "chat.model_unavailable",
            provider=chat_model.provider,
            model=chat_model.model,
            error_type=type(exc).__name__,
            status_code=getattr(exc, "status_code", None),
            error_message=str(exc)[:500],
        )
        await chat_store.append_message(
            client,
            user_id=user_id,
            thread_id=thread_id,
            role="assistant",
            content=message,
        )
        yield format_progress_event("Answer ready.", phase="complete")
        await asyncio.sleep(0)
        async for event in stream_ui_message_text(
            message,
            message_id=message_id,
            include_start=False,
        ):
            yield event
        _log_turn_complete(
            outcome="model_unavailable",
            provider=chat_model.provider,
            model=chat_model.model,
            started_at=turn_started,
            usage=usage,
        )
        return
    except Exception as exc:
        logger.exception(
            "chat.turn_failed",
            provider=chat_model.provider,
            model=chat_model.model,
            error_type=type(exc).__name__,
        )
        await chat_store.append_message(
            client,
            user_id=user_id,
            thread_id=thread_id,
            role="assistant",
            content=TURN_FAILED_MESSAGE,
        )
        yield format_progress_event("Answer ready.", phase="complete")
        await asyncio.sleep(0)
        async for event in stream_ui_message_text(
            TURN_FAILED_MESSAGE,
            message_id=message_id,
            include_start=False,
        ):
            yield event
        _log_turn_complete(
            outcome="turn_failed",
            provider=chat_model.provider,
            model=chat_model.model,
            started_at=turn_started,
            error_type=type(exc).__name__,
            usage=usage,
        )
        return

    validate_id = activity.start("validate", "Validating sources...")
    for event in emit_activity_updates():
        yield event
    await asyncio.sleep(0)

    grounding_error: GroundingError | None = None
    try:
        answer = _finalize_grounded_answer(
            answer,
            retrieved_passages,
            grounding_validator,
        )
    except GroundingError as exc:
        grounding_error = exc
        if (
            answer.citations
            and retrieved_passages
            and evidence is not None
            and usage is not None
        ):
            logger.info(
                "chat.grounding_correction",
                reason=str(exc),
                retrieved_passage_count=len(retrieved_passages),
                citation_count=len(answer.citations),
                provider=chat_model.provider,
                model=chat_model.model,
            )
            try:
                draft = await _run_citation_correction(
                    user_text=user_text,
                    failed_draft_answer=answer.answer,
                    grounding_error=str(exc),
                    evidence=evidence,
                    chat_model=chat_model,
                    generation=generation,
                    usage=usage,
                    user_id=user_id,
                    thread_id=thread_id,
                )
                answer = finalize_grounded_draft(draft, evidence)
                retrieved_passages = evidence.all_passages()
                answer = _finalize_grounded_answer(
                    answer,
                    retrieved_passages,
                    grounding_validator,
                )
                grounding_error = None
            except GroundingError as retry_exc:
                grounding_error = retry_exc

    if grounding_error is not None:
        logger.warning(
            "chat.grounding_failed",
            reason=str(grounding_error),
            retrieved_passage_count=len(retrieved_passages),
            citation_count=len(answer.citations),
            provider=chat_model.provider,
            model=chat_model.model,
        )
        activity.end(validate_id, kind="validate", label="Validation failed")
        for event in emit_activity_updates():
            yield event
            await asyncio.sleep(0)
        await chat_store.append_message(
            client,
            user_id=user_id,
            thread_id=thread_id,
            role="assistant",
            content=REFUSAL_MESSAGE,
        )
        yield format_progress_event("Answer ready.", phase="complete")
        await asyncio.sleep(0)
        async for event in stream_ui_message_text(
            REFUSAL_MESSAGE,
            message_id=message_id,
            include_start=False,
        ):
            yield event
        _log_turn_complete(
            outcome="grounding_refusal",
            provider=chat_model.provider,
            model=chat_model.model,
            started_at=turn_started,
            usage=usage,
        )
        return

    activity.end(validate_id, kind="validate", label="Sources validated")
    save_id = activity.start("save", "Saving answer...")
    for event in emit_activity_updates():
        yield event
    await asyncio.sleep(0)

    await _persist_assistant_answer(
        client,
        user_id=user_id,
        thread_id=thread_id,
        answer=answer,
        activity_log=activity_log,
    )
    activity.end(save_id, kind="save", label="Answer saved")
    for event in emit_activity_updates():
        yield event
    await asyncio.sleep(0)

    async for event in stream_grounded_answer_events(
        answer,
        message_id=message_id,
        include_start=False,
    ):
        yield event

    _log_turn_complete(
        outcome="answered",
        provider=chat_model.provider,
        model=chat_model.model,
        started_at=turn_started,
        citation_count=len(answer.citations),
        usage=usage,
    )


def _finalize_grounded_answer(
    answer: GroundedAnswer,
    retrieved_passages: list[SourcePassage],
    grounding_validator: GroundingValidator,
) -> GroundedAnswer:
    answer = repair_grounded_answer(answer, retrieved_passages)
    grounding_validator.validate(answer, retrieved_passages)
    return answer


async def _run_agent(
    user_text: str,
    *,
    user_id: UUID,
    thread_id: UUID,
    chat_model: ResolvedChatModel,
    generation: ChatGenerationConfig,
    grounding_validator: GroundingValidator,
    activity: TurnActivityEmitter | None = None,
    retriever: DocumentRetriever | None = None,
) -> tuple[GroundedAnswer, list[SourcePassage], EvidenceRegistry, TurnUsage]:
    if activity is not None:
        activity.start_thinking(f"Thinking with {chat_model.model}...")

    inner = retriever if retriever is not None else SessionPerCallDocumentRetriever()
    answer, passages, evidence, usage = await _run_agent_with_retriever(
        user_text,
        user_id=user_id,
        thread_id=thread_id,
        chat_model=chat_model,
        generation=generation,
        grounding_validator=grounding_validator,
        retriever=inner,
        activity=activity,
    )
    if activity is not None:
        activity.end_thinking()
    return answer, passages, evidence, usage


async def _run_agent_with_retriever(
    user_text: str,
    *,
    user_id: UUID,
    thread_id: UUID,
    chat_model: ResolvedChatModel,
    generation: ChatGenerationConfig,
    grounding_validator: GroundingValidator,
    retriever: DocumentRetriever,
    activity: TurnActivityEmitter | None,
) -> tuple[GroundedAnswer, list[SourcePassage], EvidenceRegistry, TurnUsage]:
    recording_retriever = _RecordingRetriever(
        inner=retriever,
        activity=activity,
    )
    budget: TurnBudget = DEFAULT_TURN_BUDGET
    evidence = EvidenceRegistry(max_passages=budget.max_unique_passages)
    usage = TurnUsage()
    deps = DocumentAgentDeps(
        user_id=user_id,
        thread_id=thread_id,
        retriever=recording_retriever,
        grounding_validator=grounding_validator,
        evidence=evidence,
        usage=usage,
        budget=budget,
    )
    agent_model = build_document_agent_model(chat_model.provider, chat_model.model)
    model_settings = build_model_settings(
        generation,
        max_tokens=budget.extractor_output_tokens,
    )

    async def event_handler(ctx, events):
        if activity is None:
            async for _event in events:
                pass
            return
        await agent_event_stream_handler(
            activity,
            model_name=chat_model.model,
            _ctx=ctx,
            events=events,
        )

    with document_agent.override(model=agent_model):
        run = await document_agent.run(
            user_text,
            deps=deps,
            model_settings=model_settings,
            event_stream_handler=event_handler if activity is not None else None,
        )
    usage.add_model_usage(
        stage="synthesis",
        model=chat_model.model,
        **_token_usage_fields(run),
    )
    logger.info(
        "chat.agent_complete",
        provider=chat_model.provider,
        model=chat_model.model,
        retrieved_passage_count=len(recording_retriever.retrieved_passages),
        **usage.as_log_fields(),
    )
    draft = run.output
    answer = finalize_grounded_draft(draft, evidence)
    return answer, recording_retriever.retrieved_passages, evidence, usage


async def _run_citation_correction(
    *,
    user_text: str,
    failed_draft_answer: str,
    grounding_error: str,
    evidence: EvidenceRegistry,
    chat_model: ResolvedChatModel,
    generation: ChatGenerationConfig,
    usage: TurnUsage,
    user_id: UUID,
    thread_id: UUID,
) -> GroundedDraft:
    """Run a no-retrieval correction pass to fix citations."""

    # Build a compact dump of existing evidence keyed by alias, using the
    # already-truncated content intended for model consumption.
    compact_rows = [row.model_dump() for row in evidence.compact_dump()]
    evidence_json = json.dumps(compact_rows, ensure_ascii=False)

    correction_prompt = (
        f"{user_text}\n\n---\n"
        "Your previous answer failed grounding validation.\n"
        f"Error: {grounding_error}\n\n"
        "Here is your previous answer:\n"
        f"{failed_draft_answer}\n\n"
        "Here is the evidence you may cite, keyed by alias:\n"
        f"{evidence_json}\n\n"
        "Rewrite the answer so that every sentence containing $, %, or numeric amounts "
        "has an inline [n] citation marker whose record matches one of the aliases "
        "above. Do not call tools or retrieve new documents. Only adjust wording and "
        "citations using this fixed evidence set."
    )

    # Disable additional retrieval during correction.
    class _DisabledRetriever:
        def search_filings(self, query: str, *, limit: int = 10) -> RetrievalResult:
            raise RuntimeError("retrieval disabled during correction")

        def search_filings_batch(
            self,
            queries: list[str],
            *,
            limit_per_query: int = 5,
        ) -> list[SourcePassage]:
            raise RuntimeError("retrieval disabled during correction")

        def read_chunk(self, chunk_id: UUID) -> SourcePassage:
            raise RuntimeError("retrieval disabled during correction")

        def read_surrounding_chunks(
            self,
            chunk_id: UUID,
            *,
            window: int = 1,
        ) -> list[SourcePassage]:
            raise RuntimeError("retrieval disabled during correction")

    class _NoOpValidator:
        def validate(
            self,
            answer: GroundedAnswer,
            retrieved_passages: list[SourcePassage],
        ) -> None:
            return None

    budget = TurnBudget(
        max_searches=0,
        max_reserve_searches=DEFAULT_TURN_BUDGET.max_reserve_searches,
        max_hits_per_search=DEFAULT_TURN_BUDGET.max_hits_per_search,
        max_unique_passages=DEFAULT_TURN_BUDGET.max_unique_passages,
        correction_output_tokens=DEFAULT_TURN_BUDGET.correction_output_tokens,
        max_corrections=DEFAULT_TURN_BUDGET.max_corrections,
    )
    deps = DocumentAgentDeps(
        user_id=user_id,
        thread_id=thread_id,
        retriever=_DisabledRetriever(),
        grounding_validator=_NoOpValidator(),
        evidence=evidence,
        usage=usage,
        budget=budget,
        correction_mode=True,
    )

    agent_model = build_document_agent_model(chat_model.provider, chat_model.model)
    model_settings = build_model_settings(
        generation,
        max_tokens=budget.correction_output_tokens,
    )

    with document_agent.override(model=agent_model):
        run = await document_agent.run(
            correction_prompt,
            deps=deps,
            model_settings=model_settings,
            event_stream_handler=None,
        )

    usage.record_correction()
    usage.add_model_usage(
        stage="correction",
        model=chat_model.model,
        **_token_usage_fields(run),
    )

    logger.info(
        "chat.agent_complete",
        provider=chat_model.provider,
        model=chat_model.model,
        retrieved_passage_count=len(evidence.all_passages()),
        correction=True,
        **usage.as_log_fields(),
    )

    return run.output


async def _persist_assistant_answer(
    client: AsyncClient,
    *,
    user_id: UUID,
    thread_id: UUID,
    answer: GroundedAnswer,
    activity_log: list[TurnActivityData] | None = None,
) -> None:
    message = await chat_store.append_message(
        client,
        user_id=user_id,
        thread_id=thread_id,
        role="assistant",
        content=answer.answer,
    )
    await chat_store.attach_citations(
        client,
        message_id=message.id,
        citations=[
            AttachCitationInput(
                chunk_id=citation.chunk_id,
                citation_index=citation.citation_index,
                excerpt=citation.excerpt,
            )
            for citation in answer.citations
        ],
    )
    activity_steps = None
    if activity_log:
        activity_steps = group_activity_steps(merge_activity_log(activity_log))
    await chat_store.update_message_data(
        client,
        message_id=message.id,
        message_data=assistant_answer_to_wire(
            answer,
            message_id=str(message.id),
            activity_steps=activity_steps,
        ),
    )

"""Coordinates one chat turn end-to-end.

Turn lifecycle: stream start -> run agent with progress -> validate -> persist -> stream answer.

The retriever and grounding validator are injected so the orchestrator stays
testable without a live LLM or database. Production turns open their own DB
session inside the background agent task so streaming stays concurrent-safe.
"""

from __future__ import annotations

import asyncio
import queue
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from uuid import UUID

import structlog
from fastapi.responses import StreamingResponse
from supabase import AsyncClient

from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from app.assistant.agent import build_document_agent_model, document_agent
from app.assistant.deps import DocumentAgentDeps, DocumentRetriever, GroundingValidator
from app.assistant.outputs import GroundedAnswer
from app.chat.generation import (
    ChatGenerationConfig,
    build_length_instruction,
    build_model_settings,
    limit_grounded_answer,
)
from app.chat.models_catalog import ResolvedChatModel
from app.chat.persistence import assistant_answer_to_wire
from app.chat.streaming import (
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
from app.grounding.validator import GroundingError, validate
from app.retrieval.document_retriever import SessionPerCallDocumentRetriever
from app.retrieval.types import RetrievalResult, SourcePassage

REFUSAL_MESSAGE = "This corpus doesn't contain enough evidence to answer that."

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


@dataclass
class _RecordingRetriever:
    """Wraps a retriever to record every passage returned during the turn."""

    inner: DocumentRetriever
    emit_progress: Callable[[str], None] | None = None
    seen: dict[UUID, SourcePassage] = field(default_factory=dict)

    def search_filings(self, query: str, *, limit: int = 10) -> RetrievalResult:
        if self.emit_progress is not None:
            self.emit_progress("Searching indexed filings...")
        result = self.inner.search_filings(query, limit=limit)
        self._record(result.passages)
        if self.emit_progress is not None:
            self.emit_progress("Analyzing retrieved passages...")
        return result

    def read_chunk(self, chunk_id: UUID) -> SourcePassage:
        if self.emit_progress is not None:
            self.emit_progress("Reading filing passage...")
        passage = self.inner.read_chunk(chunk_id)
        self._record([passage])
        if self.emit_progress is not None:
            self.emit_progress("Analyzing retrieved passages...")
        return passage

    def read_surrounding_chunks(
        self,
        chunk_id: UUID,
        *,
        window: int = 1,
    ) -> list[SourcePassage]:
        if self.emit_progress is not None:
            self.emit_progress("Reading surrounding context...")
        passages = self.inner.read_surrounding_chunks(chunk_id, window=window)
        self._record(passages)
        if self.emit_progress is not None:
            self.emit_progress("Analyzing retrieved passages...")
        return passages

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
    message_id = f"msg_{uuid.uuid4().hex}"
    progress_updates: queue.Queue[str] = queue.Queue()
    last_progress = "Analyzing your question..."

    def emit_progress(label: str) -> None:
        progress_updates.put(label)

    yield format_ui_message_sse_event({"type": "start", "messageId": message_id})
    yield format_progress_event(last_progress)
    await asyncio.sleep(0)

    agent_task = asyncio.create_task(
        _run_agent(
            user_text,
            user_id=user_id,
            thread_id=thread_id,
            chat_model=chat_model,
            generation=generation,
            grounding_validator=grounding_validator,
            on_start=emit_progress,
            retriever=retriever,
        )
    )

    try:
        while not agent_task.done():
            progress_changed = False
            while not progress_updates.empty():
                last_progress = progress_updates.get_nowait()
                progress_changed = True

            if progress_changed:
                yield format_progress_event(last_progress)
                await asyncio.sleep(0)

            await asyncio.sleep(0.05)

        while not progress_updates.empty():
            last_progress = progress_updates.get_nowait()
            yield format_progress_event(last_progress)
            await asyncio.sleep(0)

        answer, retrieved_passages = agent_task.result()
    except (ModelHTTPError, UnexpectedModelBehavior) as exc:
        message = model_unavailable_message(exc, provider=chat_model.provider)
        logger.warning(
            "chat.model_unavailable",
            provider=chat_model.provider,
            model=chat_model.model,
            error_type=type(exc).__name__,
            status_code=getattr(exc, "status_code", None),
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
        return

    yield format_progress_event("Validating sources...", phase="running")
    await asyncio.sleep(0)

    try:
        validate(answer, retrieved_passages)
    except GroundingError:
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
        return

    yield format_progress_event("Saving answer...", phase="running")
    await asyncio.sleep(0)

    await _persist_assistant_answer(
        client,
        user_id=user_id,
        thread_id=thread_id,
        answer=answer,
    )

    async for event in stream_grounded_answer_events(
        answer,
        message_id=message_id,
        include_start=False,
    ):
        yield event


async def _run_agent(
    user_text: str,
    *,
    user_id: UUID,
    thread_id: UUID,
    chat_model: ResolvedChatModel,
    generation: ChatGenerationConfig,
    grounding_validator: GroundingValidator,
    on_start: Callable[[str], None] | None = None,
    retriever: DocumentRetriever | None = None,
) -> tuple[GroundedAnswer, list[SourcePassage]]:
    if on_start is not None:
        on_start(f"Thinking with {chat_model.model}...")

    inner = retriever if retriever is not None else SessionPerCallDocumentRetriever()
    return await _run_agent_with_retriever(
        user_text,
        user_id=user_id,
        thread_id=thread_id,
        chat_model=chat_model,
        generation=generation,
        grounding_validator=grounding_validator,
        retriever=inner,
        on_start=on_start,
    )


async def _run_agent_with_retriever(
    user_text: str,
    *,
    user_id: UUID,
    thread_id: UUID,
    chat_model: ResolvedChatModel,
    generation: ChatGenerationConfig,
    grounding_validator: GroundingValidator,
    retriever: DocumentRetriever,
    on_start: Callable[[str], None] | None,
) -> tuple[GroundedAnswer, list[SourcePassage]]:
    recording_retriever = _RecordingRetriever(
        inner=retriever,
        emit_progress=on_start,
    )
    deps = DocumentAgentDeps(
        user_id=user_id,
        thread_id=thread_id,
        retriever=recording_retriever,
        grounding_validator=grounding_validator,
    )
    agent_model = build_document_agent_model(chat_model.provider, chat_model.model)
    model_settings = build_model_settings(generation)
    prompt = user_text
    if generation.max_output_tokens is not None:
        prompt = user_text + build_length_instruction(generation.max_output_tokens)
    with document_agent.override(model=agent_model):
        run = await document_agent.run(prompt, deps=deps, model_settings=model_settings)
    return limit_grounded_answer(run.output, generation.max_output_tokens), recording_retriever.retrieved_passages


async def _persist_assistant_answer(
    client: AsyncClient,
    *,
    user_id: UUID,
    thread_id: UUID,
    answer: GroundedAnswer,
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
    await chat_store.update_message_data(
        client,
        message_id=message.id,
        message_data=assistant_answer_to_wire(answer, message_id=str(message.id)),
    )

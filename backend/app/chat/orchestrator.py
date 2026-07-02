"""Coordinates one chat turn end-to-end.

Turn lifecycle: build deps -> invoke agent -> validate grounding -> persist -> stream.

The retriever and grounding validator are injected so the orchestrator stays
testable without a live LLM or database. The concrete retriever adapter and
grounding validator are wired at the API boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from fastapi.responses import StreamingResponse
from supabase import AsyncClient

from app.assistant.agent import document_agent
from app.assistant.deps import DocumentAgentDeps, DocumentRetriever, GroundingValidator
from app.assistant.outputs import GroundedAnswer
from app.chat.streaming import stream_grounded_answer, stream_refusal
from app.database import chats as chat_store
from app.database.chats import AttachCitationInput
from app.retrieval.types import RetrievalResult, SourcePassage

REFUSAL_MESSAGE = "This corpus doesn't contain enough evidence to answer that."

__all__ = ["REFUSAL_MESSAGE", "GroundingError", "run_chat_turn"]


class GroundingError(Exception):
    """Raised by a grounding validator when an answer fails policy checks."""


@dataclass
class _RecordingRetriever:
    """Wraps a retriever to record every passage returned during the turn.

    Grounding requires knowing exactly which passages were retrieved for this
    request, so the validator can reject citations to chunks that were never
    surfaced to the model.
    """

    inner: DocumentRetriever
    seen: dict[UUID, SourcePassage] = field(default_factory=dict)

    def search_filings(self, query: str, *, limit: int = 10) -> RetrievalResult:
        result = self.inner.search_filings(query, limit=limit)
        self._record(result.passages)
        return result

    def read_chunk(self, chunk_id: UUID) -> SourcePassage:
        passage = self.inner.read_chunk(chunk_id)
        self._record([passage])
        return passage

    def read_surrounding_chunks(
        self,
        chunk_id: UUID,
        *,
        window: int = 1,
    ) -> list[SourcePassage]:
        passages = self.inner.read_surrounding_chunks(chunk_id, window=window)
        self._record(passages)
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
    retriever: DocumentRetriever,
    grounding_validator: GroundingValidator,
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

    recording_retriever = _RecordingRetriever(inner=retriever)
    deps = DocumentAgentDeps(
        user_id=user_id,
        thread_id=thread_id,
        retriever=recording_retriever,
        grounding_validator=grounding_validator,
    )

    run = await document_agent.run(user_text, deps=deps)
    answer = run.output

    try:
        grounding_validator.validate(answer, recording_retriever.retrieved_passages)
    except GroundingError:
        await chat_store.append_message(
            client,
            user_id=user_id,
            thread_id=thread_id,
            role="assistant",
            content=REFUSAL_MESSAGE,
        )
        return stream_refusal(REFUSAL_MESSAGE)

    await _persist_assistant_answer(
        client,
        user_id=user_id,
        thread_id=thread_id,
        answer=answer,
    )
    return stream_grounded_answer(answer)


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

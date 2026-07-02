"""Rebuild persisted assistant UI messages from stored rows."""

from __future__ import annotations

from uuid import UUID

from app.assistant.outputs import Citation, GroundedAnswer
from app.chat.messages import grounded_answer_to_ui_message, ui_message_to_wire
from app.database.chats import ChatMessageRecord, MessageCitationRecord
from app.database.session import session_scope
from app.retrieval.retriever import _load_chunks, _to_source_passage
from app.retrieval.types import SourcePassage


def assistant_answer_to_wire(answer: GroundedAnswer, *, message_id: str) -> dict:
    ui_message = grounded_answer_to_ui_message(answer, message_id=message_id)
    return ui_message_to_wire(ui_message)


def enrich_assistant_messages(
    messages: list[ChatMessageRecord],
    citations_by_message: dict[UUID, list[MessageCitationRecord]],
) -> list[ChatMessageRecord]:
    assistant_ids = [
        message.id
        for message in messages
        if message.role == "assistant" and message.message_data is None
    ]
    if not assistant_ids:
        return messages

    chunk_ids = {
        citation.chunk_id
        for message_id in assistant_ids
        for citation in citations_by_message.get(message_id, [])
    }
    passages_by_chunk = _load_passages_by_chunk(chunk_ids)

    enriched: list[ChatMessageRecord] = []
    for message in messages:
        if message.role != "assistant" or message.message_data is not None:
            enriched.append(message)
            continue

        citations = citations_by_message.get(message.id, [])
        if not citations:
            enriched.append(message)
            continue

        grounded = GroundedAnswer(
            answer=message.content,
            citations=[
                Citation(
                    citation_index=citation.citation_index,
                    chunk_id=citation.chunk_id,
                    excerpt=citation.excerpt or "",
                )
                for citation in citations
            ],
            cited_passages=[
                passages_by_chunk[citation.chunk_id]
                for citation in citations
                if citation.chunk_id in passages_by_chunk
            ],
        )
        enriched.append(
            message.model_copy(
                update={
                    "message_data": assistant_answer_to_wire(
                        grounded,
                        message_id=str(message.id),
                    )
                }
            )
        )

    return enriched


def _load_passages_by_chunk(chunk_ids: set[UUID]) -> dict[UUID, SourcePassage]:
    if not chunk_ids:
        return {}

    with session_scope() as session:
        chunks = _load_chunks(session, list(chunk_ids))
        return {
            chunk_id: _to_source_passage(chunk, score=0.0)
            for chunk_id, chunk in chunks.items()
        }

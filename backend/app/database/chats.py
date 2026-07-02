"""Typed helpers for chat thread and message persistence."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel
from supabase import AsyncClient

from app.database.supabase import create_admin_client


class ChatThreadRecord(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageRecord(BaseModel):
    id: UUID
    thread_id: UUID
    role: str
    content: str
    message_data: dict | None = None
    created_at: datetime


class MessageCitationRecord(BaseModel):
    id: UUID
    message_id: UUID
    chunk_id: UUID
    citation_index: int
    excerpt: str | None = None
    created_at: datetime


class AttachCitationInput(BaseModel):
    chunk_id: UUID
    citation_index: int
    excerpt: str | None = None


class ChatNotFoundError(Exception):
    """Raised when a thread does not exist."""


class ChatForbiddenError(Exception):
    """Raised when a thread belongs to another user."""


async def list_threads(client: AsyncClient, user_id: UUID) -> list[ChatThreadRecord]:
    response = (
        await client.table("chat_threads")
        .select("*")
        .eq("user_id", str(user_id))
        .order("updated_at", desc=True)
        .execute()
    )
    return [_parse_thread(row) for row in response.data or []]


async def create_thread(client: AsyncClient, user_id: UUID, title: str) -> ChatThreadRecord:
    response = (
        await client.table("chat_threads")
        .insert({"user_id": str(user_id), "title": title})
        .select("*")
        .single()
        .execute()
    )
    return _parse_thread(response.data)


async def get_thread(client: AsyncClient, user_id: UUID, thread_id: UUID) -> ChatThreadRecord:
    response = (
        await client.table("chat_threads")
        .select("*")
        .eq("id", str(thread_id))
        .maybe_single()
        .execute()
    )
    if response.data is not None:
        thread = _parse_thread(response.data)
        if thread.user_id != user_id:
            raise ChatForbiddenError(str(thread_id))
        return thread

    admin = await create_admin_client()
    exists = (
        await admin.table("chat_threads")
        .select("user_id")
        .eq("id", str(thread_id))
        .maybe_single()
        .execute()
    )
    if exists.data is not None:
        raise ChatForbiddenError(str(thread_id))
    raise ChatNotFoundError(str(thread_id))


async def list_thread_messages(
    client: AsyncClient,
    user_id: UUID,
    thread_id: UUID,
) -> list[ChatMessageRecord]:
    await get_thread(client, user_id, thread_id)
    response = (
        await client.table("chat_messages")
        .select("*")
        .eq("thread_id", str(thread_id))
        .order("created_at")
        .execute()
    )
    return [_parse_message(row) for row in response.data or []]


async def append_message(
    client: AsyncClient,
    *,
    user_id: UUID,
    thread_id: UUID,
    role: str,
    content: str,
    message_data: dict | None = None,
) -> ChatMessageRecord:
    await get_thread(client, user_id, thread_id)
    payload: dict[str, object] = {
        "thread_id": str(thread_id),
        "role": role,
        "content": content,
    }
    if message_data is not None:
        payload["message_data"] = message_data

    response = await client.table("chat_messages").insert(payload).select("*").single().execute()
    await client.table("chat_threads").update({"updated_at": datetime.now().isoformat()}).eq(
        "id",
        str(thread_id),
    ).execute()
    return _parse_message(response.data)


async def attach_citations(
    client: AsyncClient,
    *,
    message_id: UUID,
    citations: list[AttachCitationInput],
) -> list[MessageCitationRecord]:
    if not citations:
        return []

    rows = [
        {
            "message_id": str(message_id),
            "chunk_id": str(citation.chunk_id),
            "citation_index": citation.citation_index,
            "excerpt": citation.excerpt,
        }
        for citation in citations
    ]
    response = await client.table("message_citations").insert(rows).select("*").execute()
    return [_parse_citation(row) for row in response.data or []]


def _parse_thread(row: dict) -> ChatThreadRecord:
    return ChatThreadRecord(
        id=UUID(row["id"]),
        user_id=UUID(row["user_id"]),
        title=row["title"],
        created_at=_parse_timestamp(row["created_at"]),
        updated_at=_parse_timestamp(row["updated_at"]),
    )


def _parse_message(row: dict) -> ChatMessageRecord:
    return ChatMessageRecord(
        id=UUID(row["id"]),
        thread_id=UUID(row["thread_id"]),
        role=row["role"],
        content=row["content"],
        message_data=row.get("message_data"),
        created_at=_parse_timestamp(row["created_at"]),
    )


def _parse_citation(row: dict) -> MessageCitationRecord:
    return MessageCitationRecord(
        id=UUID(row["id"]),
        message_id=UUID(row["message_id"]),
        chunk_id=UUID(row["chunk_id"]),
        citation_index=row["citation_index"],
        excerpt=row.get("excerpt"),
        created_at=_parse_timestamp(row["created_at"]),
    )


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

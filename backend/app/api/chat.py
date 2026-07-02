"""Chat thread REST and streaming routes."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from supabase import AsyncClient

from app.auth.dependencies import get_current_user, get_user_client
from app.auth.types import CurrentUser
from app.chat.messages import extract_latest_user_text
from app.chat.orchestrator import run_chat_turn
from app.database import chats as chat_store
from app.database.chats import ChatForbiddenError, ChatNotFoundError
from app.database.session import session_scope
from app.grounding.validator import grounding_validator
from app.retrieval.document_retriever import SessionDocumentRetriever

router = APIRouter(tags=["chat"])


class ThreadSummary(BaseModel):
    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime


class CreateThreadRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class MessageSummary(BaseModel):
    id: UUID
    role: str
    content: str
    created_at: datetime
    message_data: dict | None = None


class StreamChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    thread_id: UUID = Field(alias="threadId")
    messages: list[dict]


@router.get("/threads", response_model=list[ThreadSummary])
async def list_threads(
    user: CurrentUser = Depends(get_current_user),
    client: AsyncClient = Depends(get_user_client),
) -> list[ThreadSummary]:
    threads = await chat_store.list_threads(client, user.id)
    return [
        ThreadSummary(
            id=thread.id,
            title=thread.title,
            created_at=thread.created_at,
            updated_at=thread.updated_at,
        )
        for thread in threads
    ]


@router.post("/threads", response_model=ThreadSummary, status_code=status.HTTP_201_CREATED)
async def create_thread(
    body: CreateThreadRequest,
    user: CurrentUser = Depends(get_current_user),
    client: AsyncClient = Depends(get_user_client),
) -> ThreadSummary:
    thread = await chat_store.create_thread(client, user.id, body.title)
    return ThreadSummary(
        id=thread.id,
        title=thread.title,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
    )


@router.get("/threads/{thread_id}/messages", response_model=list[MessageSummary])
async def list_thread_messages(
    thread_id: UUID,
    user: CurrentUser = Depends(get_current_user),
    client: AsyncClient = Depends(get_user_client),
) -> list[MessageSummary]:
    try:
        messages = await chat_store.list_thread_messages(client, user.id, thread_id)
    except ChatForbiddenError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden") from exc
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found") from exc

    return [
        MessageSummary(
            id=message.id,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
            message_data=message.message_data,
        )
        for message in messages
    ]


@router.post("/chat/stream")
async def stream_chat(
    body: StreamChatRequest,
    user: CurrentUser = Depends(get_current_user),
    client: AsyncClient = Depends(get_user_client),
):
    try:
        await chat_store.get_thread(client, user.id, body.thread_id)
    except ChatForbiddenError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden") from exc
    except ChatNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found") from exc

    try:
        user_text = extract_latest_user_text(body.messages)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    with session_scope() as session:
        retriever = SessionDocumentRetriever(session)
        return await run_chat_turn(
            client,
            user_id=user.id,
            thread_id=body.thread_id,
            user_text=user_text,
            user_message_data=body.messages[-1] if body.messages else None,
            retriever=retriever,
            grounding_validator=grounding_validator,
        )

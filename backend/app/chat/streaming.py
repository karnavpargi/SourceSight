"""AI SDK-compatible streaming responses."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse

from app.assistant.outputs import GroundedAnswer
from app.chat.messages import (
    ChatUIMessage,
    TextPart,
    grounded_answer_to_ui_message,
)

UI_MESSAGE_STREAM_VERSION = "v1"
UI_MESSAGE_STREAM_HEADER = "x-vercel-ai-ui-message-stream"

STUB_ASSISTANT_REPLY = (
    "SourceSight stub: your message was saved. Retrieval and grounded answers are not wired yet."
)


def format_ui_message_sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


async def stream_ui_message_text(text: str) -> AsyncIterator[str]:
    """Emit a minimal AI SDK UI message stream for plain assistant text."""
    message_id = f"msg_{uuid.uuid4().hex}"
    ui_message = ChatUIMessage(id=message_id, role="assistant", parts=[TextPart(text=text)])
    async for event in stream_ui_message(ui_message):
        yield event


async def stream_ui_message(message: ChatUIMessage) -> AsyncIterator[str]:
    """Emit an AI SDK UI message stream for a complete assistant message."""
    yield format_ui_message_sse_event({"type": "start", "messageId": message.id})

    for part in message.parts:
        if isinstance(part, TextPart):
            async for event in _stream_text_part(part.text):
                yield event
            continue

        yield format_ui_message_sse_event(part.model_dump(mode="json", exclude_none=True))

    yield format_ui_message_sse_event({"type": "finish"})
    yield "data: [DONE]\n\n"


async def _stream_text_part(text: str) -> AsyncIterator[str]:
    text_id = f"text_{uuid.uuid4().hex}"
    yield format_ui_message_sse_event({"type": "text-start", "id": text_id})

    for chunk in _chunk_text(text):
        yield format_ui_message_sse_event({"type": "text-delta", "id": text_id, "delta": chunk})
        await asyncio.sleep(0)

    yield format_ui_message_sse_event({"type": "text-end", "id": text_id})


def ui_message_stream_response(text: str = STUB_ASSISTANT_REPLY) -> StreamingResponse:
    return _stream_response(stream_ui_message_text(text))


def stream_refusal(text: str) -> StreamingResponse:
    """Stream a controlled refusal as plain assistant text (no citations)."""
    return _stream_response(stream_ui_message_text(text))


def stream_grounded_answer(answer: GroundedAnswer) -> StreamingResponse:
    """Stream a grounded answer with text, citation, and source-passage parts."""
    message_id = f"msg_{uuid.uuid4().hex}"
    ui_message = grounded_answer_to_ui_message(answer, message_id=message_id)
    return _stream_response(stream_ui_message(ui_message))


def _stream_response(stream: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            UI_MESSAGE_STREAM_HEADER: UI_MESSAGE_STREAM_VERSION,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _chunk_text(text: str, *, chunk_size: int = 24) -> list[str]:
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]

"""AI SDK-compatible streaming responses."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse

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
    text_id = f"text_{uuid.uuid4().hex}"

    yield format_ui_message_sse_event({"type": "start", "messageId": message_id})
    yield format_ui_message_sse_event({"type": "text-start", "id": text_id})

    for chunk in _chunk_text(text):
        yield format_ui_message_sse_event({"type": "text-delta", "id": text_id, "delta": chunk})
        await asyncio.sleep(0)

    yield format_ui_message_sse_event({"type": "text-end", "id": text_id})
    yield format_ui_message_sse_event({"type": "finish"})
    yield "data: [DONE]\n\n"


def ui_message_stream_response(text: str = STUB_ASSISTANT_REPLY) -> StreamingResponse:
    return StreamingResponse(
        stream_ui_message_text(text),
        media_type="text/event-stream",
        headers={
            UI_MESSAGE_STREAM_HEADER: UI_MESSAGE_STREAM_VERSION,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


def _chunk_text(text: str, *, chunk_size: int = 24) -> list[str]:
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]

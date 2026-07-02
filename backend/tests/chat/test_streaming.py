import asyncio

from app.chat.streaming import STUB_ASSISTANT_REPLY, format_ui_message_sse_event, stream_ui_message_text


def test_stream_ui_message_text_emits_ai_sdk_events() -> None:
    async def collect() -> list[str]:
        return [chunk async for chunk in stream_ui_message_text("Hi there")]

    chunks = asyncio.run(collect())
    joined = "".join(chunks)

    assert '"type":"start"' in joined.replace(" ", "")
    assert '"type":"text-delta"' in joined.replace(" ", "")
    assert '"type":"finish"' in joined.replace(" ", "")
    assert "data: [DONE]" in joined
    assert "Hi there" in joined


def test_format_ui_message_sse_event() -> None:
    event = format_ui_message_sse_event({"type": "finish"})
    assert event == 'data: {"type":"finish"}\n\n'


def test_stub_reply_is_non_empty() -> None:
    assert STUB_ASSISTANT_REPLY.strip()

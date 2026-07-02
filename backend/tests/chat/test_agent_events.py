from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ToolCallPart,
    ToolReturnPart,
)

from app.chat.agent_events import bridge_agent_stream_events
from app.chat.turn_activity import TurnActivityEmitter


@pytest.mark.anyio
async def test_bridge_emits_tool_lifecycle() -> None:
    tool_call = ToolCallPart(
        tool_name="read_chunk",
        args={"chunk_id": "11111111-1111-1111-1111-111111111111"},
        tool_call_id="call-2",
    )
    tool_result = ToolReturnPart(
        tool_name="read_chunk",
        content="passage",
        tool_call_id="call-2",
    )

    async def events():
        yield FunctionToolCallEvent(part=tool_call)
        yield FunctionToolResultEvent(part=tool_result, content="passage")

    emitter = TurnActivityEmitter()
    emitter.start_thinking("Thinking with test-model...")
    await bridge_agent_stream_events(emitter, model_name="test-model", events=events())
    updates = emitter.drain()

    assert any(update.kind == "read_chunk" and update.phase == "start" for update in updates)
    assert any(update.kind == "read_chunk" and update.phase == "end" for update in updates)

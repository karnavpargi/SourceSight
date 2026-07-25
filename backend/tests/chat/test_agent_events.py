from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ToolCallPart,
    ToolReturnPart,
)

from app.chat.agent_events import (
    ROUTED_STAGE_RETRIEVE,
    ROUTED_STAGE_LABELS,
    bridge_agent_stream_events,
    start_routed_stage,
)
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


def test_start_routed_stage_uses_human_readable_labels() -> None:
    emitter = TurnActivityEmitter()
    step_id = start_routed_stage(emitter, stage=ROUTED_STAGE_RETRIEVE, bind_active=True)
    updates = emitter.drain()
    assert updates[0].phase == "start"
    assert updates[0].kind == ROUTED_STAGE_RETRIEVE
    assert updates[0].label == ROUTED_STAGE_LABELS[ROUTED_STAGE_RETRIEVE]
    # Ensure the stage can receive update_active() calls when bound.
    emitter.update_active("Searching indexed filings...", detail="NVDA")
    updates = emitter.drain()
    assert updates and updates[-1].phase == "update"
    emitter.end(step_id, kind=ROUTED_STAGE_RETRIEVE, label="Evidence retrieved")

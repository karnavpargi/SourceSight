from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ToolCallPart,
    ToolReturnPart,
)

from app.chat.agent_events import bridge_agent_stream_events, tool_detail
from app.chat.turn_activity import TurnActivityEmitter


@pytest.mark.anyio
async def test_bridge_agent_stream_events_emits_tool_start_and_end() -> None:
    tool_call = ToolCallPart(
        tool_name="search_filings",
        args={"query": "NVDA Data Center demand"},
        tool_call_id="call-1",
    )
    tool_result = ToolReturnPart(
        tool_name="search_filings",
        content="results",
        tool_call_id="call-1",
    )

    async def events():
        yield FunctionToolCallEvent(part=tool_call)
        yield FunctionToolResultEvent(part=tool_result, content="results")

    emitter = TurnActivityEmitter()
    await bridge_agent_stream_events(emitter, model_name="test-model", events=events())
    updates = emitter.drain()

    assert [update.phase for update in updates] == ["start", "end"]
    assert updates[0].kind == "search_filings"
    assert updates[0].detail == "NVDA Data Center demand"
    assert updates[1].step_id == updates[0].step_id


def test_tool_detail_formats_search_query() -> None:
    assert tool_detail("search_filings", {"query": "  revenue mix  "}) == "revenue mix"


def test_start_thinking_reuses_open_step() -> None:
    emitter = TurnActivityEmitter()
    first = emitter.start_thinking("Analyzing your question...")
    second = emitter.start_thinking("Thinking with test-model...")
    updates = emitter.drain()

    assert first == second
    assert [update.phase for update in updates] == ["start", "update"]
    assert updates[0].step_id == updates[1].step_id


def test_turn_activity_emitter_update_requires_active_tool() -> None:
    emitter = TurnActivityEmitter()
    emitter.update_active("Should not emit")
    assert emitter.drain() == []

    step_id = emitter.start("search_filings", "Searching filings")
    emitter.bind_active_tool(step_id)
    emitter.update_active("Searching indexed filings...", detail="NVDA")
    updates = emitter.drain()

    assert len(updates) == 2
    assert updates[1].phase == "update"
    assert updates[1].label == "Searching indexed filings..."


def test_recording_retriever_emits_activity_updates_during_search() -> None:
    from unittest.mock import Mock

    from app.chat.orchestrator import _RecordingRetriever
    from app.retrieval.types import RetrievalResult

    emitter = TurnActivityEmitter()
    inner = Mock()
    inner.search_filings.return_value = RetrievalResult(query="NVDA", passages=[])
    recorder = _RecordingRetriever(inner=inner, activity=emitter)

    step_id = emitter.start("search_filings", "Searching filings")
    emitter.bind_active_tool(step_id)
    recorder.search_filings("NVDA revenue", limit=5)

    updates = emitter.drain()
    assert any(update.phase == "update" for update in updates)
    assert any("indexed filings" in update.label for update in updates)

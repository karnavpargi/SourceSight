"""Map PydanticAI agent stream events to turn activity updates."""

from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any
from uuid import UUID

from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartEndEvent,
    PartStartEvent,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.tools import RunContext

from app.chat.turn_activity import TurnActivityEmitter

_RETRIEVAL_TOOLS = frozenset(
    {"search_filings", "read_chunk", "read_surrounding_chunks"},
)

_TOOL_LABELS: dict[str, str] = {
    "search_filings": "Searching filings",
    "read_chunk": "Reading passage",
    "read_surrounding_chunks": "Reading surrounding context",
}


def tool_detail(tool_name: str, args: Any) -> str | None:
    if not isinstance(args, dict):
        return None

    if tool_name == "search_filings":
        query = args.get("query")
        if isinstance(query, str) and query.strip():
            return query.strip()
        return None

    if tool_name in {"read_chunk", "read_surrounding_chunks"}:
        chunk_id = args.get("chunk_id")
        if chunk_id is None:
            return None
        try:
            chunk_label = f"Chunk {UUID(str(chunk_id))}"
        except ValueError:
            chunk_label = str(chunk_id)
        if tool_name == "read_surrounding_chunks":
            window = args.get("window")
            if window is not None:
                return f"{chunk_label} · window ±{window}"
        return chunk_label

    return None


def _is_thinking_part(part: object) -> bool:
    return isinstance(part, ThinkingPart)


async def bridge_agent_stream_events(
    emitter: TurnActivityEmitter,
    *,
    model_name: str,
    events: AsyncIterable[object],
) -> None:
    """Consume PydanticAI stream events and emit structured turn activity."""
    open_tools: dict[str, tuple[str, str]] = {}

    async for event in events:
        if isinstance(event, PartStartEvent) and _is_thinking_part(event.part):
            emitter.start_thinking(f"Thinking with {model_name}...")
            continue

        if isinstance(event, PartEndEvent) and _is_thinking_part(event.part):
            emitter.end_thinking()
            continue

        if isinstance(event, FunctionToolCallEvent):
            emitter.end_thinking()
            part = event.part
            if not isinstance(part, ToolCallPart):
                continue

            tool_name = part.tool_name
            if tool_name not in _RETRIEVAL_TOOLS:
                continue

            label = _TOOL_LABELS.get(tool_name, tool_name.replace("_", " "))
            step_id = emitter.start(
                tool_name,
                label,
                detail=tool_detail(tool_name, part.args),
                step_id=part.tool_call_id,
            )
            open_tools[part.tool_call_id] = (step_id, tool_name)
            emitter.bind_active_tool(step_id)
            continue

        if isinstance(event, FunctionToolResultEvent):
            tool_info = open_tools.pop(event.tool_call_id, None)
            if tool_info is None:
                continue
            step_id, tool_name = tool_info
            end_label = _TOOL_LABELS.get(tool_name, "Done")
            emitter.end(step_id, kind=tool_name, label=end_label)


async def agent_event_stream_handler(
    emitter: TurnActivityEmitter,
    *,
    model_name: str,
    _ctx: RunContext[Any],
    events: AsyncIterable[object],
) -> None:
    await bridge_agent_stream_events(emitter, model_name=model_name, events=events)

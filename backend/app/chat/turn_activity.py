"""Structured turn activity for chat streaming."""

from __future__ import annotations

import queue
import uuid
from dataclasses import dataclass, field

from app.chat.messages import TurnActivityData

ActivityKind = str


@dataclass
class TurnActivityEmitter:
    """Thread-safe activity queue for agent tool calls and orchestrator phases."""

    _queue: queue.Queue[TurnActivityData] = field(default_factory=queue.Queue)
    _current_tool_step_id: str | None = field(default=None, init=False)
    _open_thinking_id: str | None = field(default=None, init=False)
    _step_labels: dict[str, str] = field(default_factory=dict, init=False)
    _step_kinds: dict[str, str] = field(default_factory=dict, init=False)
    _order: int = field(default=0, init=False)

    def drain(self) -> list[TurnActivityData]:
        items: list[TurnActivityData] = []
        while not self._queue.empty():
            items.append(self._queue.get_nowait())
        return items

    def start(
        self,
        kind: ActivityKind,
        label: str,
        *,
        detail: str | None = None,
        step_id: str | None = None,
    ) -> str:
        resolved_step_id = step_id or uuid.uuid4().hex
        self._step_labels[resolved_step_id] = label
        self._step_kinds[resolved_step_id] = kind
        self._order += 1
        self._queue.put(
            TurnActivityData(
                step_id=resolved_step_id,
                kind=kind,
                phase="start",
                label=label,
                detail=detail,
                order=self._order,
            )
        )
        return resolved_step_id

    def update(
        self,
        step_id: str,
        label: str,
        *,
        kind: ActivityKind | None = None,
        detail: str | None = None,
    ) -> None:
        if label:
            self._step_labels[step_id] = label
        self._order += 1
        self._queue.put(
            TurnActivityData(
                step_id=step_id,
                kind=kind or self._step_kind(step_id),
                phase="update",
                label=label,
                detail=detail,
                order=self._order,
            )
        )

    def update_active(self, label: str, *, detail: str | None = None) -> None:
        if self._current_tool_step_id is None:
            return
        self.update(self._current_tool_step_id, label, detail=detail)

    def bind_active_tool(self, step_id: str) -> None:
        self._current_tool_step_id = step_id

    def end(
        self,
        step_id: str,
        *,
        kind: ActivityKind | None = None,
        label: str | None = None,
    ) -> None:
        resolved_label = label or self._step_labels.get(step_id, "")
        self._order += 1
        self._queue.put(
            TurnActivityData(
                step_id=step_id,
                kind=kind or self._step_kind(step_id),
                phase="end",
                label=resolved_label,
                order=self._order,
            )
        )
        if self._current_tool_step_id == step_id:
            self._current_tool_step_id = None
        if self._open_thinking_id == step_id:
            self._open_thinking_id = None
        self._step_labels.pop(step_id, None)
        self._step_kinds.pop(step_id, None)

    def start_thinking(self, label: str) -> str:
        if self._open_thinking_id is not None:
            self.update(self._open_thinking_id, label, kind="thinking")
            return self._open_thinking_id
        self._open_thinking_id = self.start("thinking", label)
        return self._open_thinking_id

    def end_thinking(self) -> None:
        if self._open_thinking_id is None:
            return
        self.end(self._open_thinking_id, kind="thinking")
        self._open_thinking_id = None

    def _step_kind(self, step_id: str) -> ActivityKind:
        return self._step_kinds.get(step_id, "analyze")

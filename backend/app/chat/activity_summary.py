"""Merge streamed activity events into persisted step snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from app.chat.messages import TurnActivityData

_KIND_LABELS: dict[str, str] = {
    "thinking": "Thinking",
    "search_filings": "Searching filings",
    "read_chunk": "Reading passage",
    "read_surrounding_chunks": "Reading surrounding context",
    "validate": "Validating sources",
    "save": "Saving answer",
}


@dataclass(frozen=True)
class _MergedStep:
    step_id: str
    kind: str
    label: str
    detail: str | None
    order: int


def merge_activity_log(events: list[TurnActivityData]) -> list[TurnActivityData]:
    merged: dict[str, _MergedStep] = {}

    for event in events:
        if not event.label.strip():
            continue

        existing = merged.get(event.step_id)
        base = existing or _MergedStep(
            step_id=event.step_id,
            kind=event.kind,
            label=event.label,
            detail=event.detail,
            order=event.order,
        )

        if event.phase == "start":
            merged[event.step_id] = _MergedStep(
                step_id=event.step_id,
                kind=event.kind,
                label=event.label,
                detail=event.detail,
                order=event.order,
            )
            continue

        if event.phase == "update":
            merged[event.step_id] = _MergedStep(
                step_id=event.step_id,
                kind=event.kind or base.kind,
                label=event.label or base.label,
                detail=event.detail if event.detail is not None else base.detail,
                order=max(base.order, event.order),
            )
            continue

        merged[event.step_id] = _MergedStep(
            step_id=event.step_id,
            kind=event.kind or base.kind,
            label=event.label or base.label,
            detail=base.detail,
            order=max(base.order, event.order),
        )

    steps = sorted(merged.values(), key=lambda step: step.order)
    return [
        TurnActivityData(
            step_id=step.step_id,
            kind=step.kind,
            phase="end",
            label=step.label,
            detail=step.detail,
            order=step.order,
        )
        for step in steps
    ]


def group_activity_steps(steps: list[TurnActivityData]) -> list[TurnActivityData]:
    """Collapse consecutive completed steps of the same kind for compact storage."""
    if not steps:
        return []

    grouped: list[TurnActivityData] = []
    for step in steps:
        previous = grouped[-1] if grouped else None
        if (
            previous is not None
            and previous.kind == step.kind
            and previous.phase == "end"
            and step.phase == "end"
        ):
            count = _group_count(previous.label) + 1
            label = _group_label(step.kind, count)
            grouped[-1] = TurnActivityData(
                step_id=previous.step_id,
                kind=step.kind,
                phase="end",
                label=label,
                detail=step.detail or previous.detail,
                order=step.order,
            )
            continue

        grouped.append(step)

    return grouped


def _group_label(kind: str, count: int) -> str:
    base = _KIND_LABELS.get(kind, kind.replace("_", " ").title())
    if count <= 1:
        return base
    return f"{base} ×{count}"


def _group_count(label: str) -> int:
    if " ×" not in label:
        return 1
    suffix = label.rsplit(" ×", maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else 1

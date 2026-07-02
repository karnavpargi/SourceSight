from app.chat.activity_summary import group_activity_steps, merge_activity_log
from app.chat.messages import TurnActivityData


def test_merge_activity_log_collapses_stream_events_per_step() -> None:
    events = [
        TurnActivityData(
            step_id="a",
            kind="search_filings",
            phase="start",
            label="Searching filings",
            detail="NVDA",
            order=1,
        ),
        TurnActivityData(
            step_id="a",
            kind="search_filings",
            phase="update",
            label="Analyzing retrieved passages...",
            detail="3 passages found",
            order=2,
        ),
        TurnActivityData(
            step_id="a",
            kind="search_filings",
            phase="end",
            label="Searching filings",
            order=3,
        ),
    ]

    merged = merge_activity_log(events)

    assert len(merged) == 1
    assert merged[0].phase == "end"
    assert merged[0].label == "Searching filings"
    assert merged[0].detail == "3 passages found"


def test_group_activity_steps_collapses_consecutive_same_kind() -> None:
    steps = [
        TurnActivityData(
            step_id="1",
            kind="search_filings",
            phase="end",
            label="Searching filings",
            order=1,
        ),
        TurnActivityData(
            step_id="2",
            kind="search_filings",
            phase="end",
            label="Searching filings",
            detail="NVDA demand",
            order=2,
        ),
        TurnActivityData(
            step_id="3",
            kind="validate",
            phase="end",
            label="Validating sources",
            order=3,
        ),
    ]

    grouped = group_activity_steps(steps)

    assert len(grouped) == 2
    assert grouped[0].label == "Searching filings ×2"
    assert grouped[0].detail == "NVDA demand"
    assert grouped[1].kind == "validate"

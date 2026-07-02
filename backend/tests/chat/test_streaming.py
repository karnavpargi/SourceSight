import asyncio
import json
from datetime import date
from uuid import UUID

from app.assistant.outputs import Citation, GroundedAnswer
from app.chat.streaming import (
    STUB_ASSISTANT_REPLY,
    format_ui_message_sse_event,
    stream_grounded_answer,
    stream_ui_message_text,
)
from app.retrieval.types import SourcePassage

CHUNK_ID = UUID("11111111-1111-1111-1111-111111111111")
DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")


def _passage() -> SourcePassage:
    return SourcePassage(
        chunk_id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        chunk_index=3,
        content="AWS operating income increased.",
        section="Item 7. MD&A",
        ticker="AMZN",
        company_name="Amazon.com, Inc.",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001018724-24-000123",
        filing_date=date(2024, 2, 2),
        source_url="https://example.com/amzn-10k",
        score=0.42,
    )


def _grounded_answer() -> GroundedAnswer:
    return GroundedAnswer(
        answer="AWS operating income rose [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=CHUNK_ID,
                excerpt="AWS operating income increased.",
            )
        ],
        cited_passages=[_passage()],
    )


def _parse_events(body: str) -> list[dict]:
    events: list[dict] = []
    for line in body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        events.append(json.loads(line[len("data: ") :]))
    return events


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


def test_stream_grounded_answer_emits_progress_and_text() -> None:
    from app.chat.streaming import format_progress_event, stream_grounded_answer_events

    async def collect() -> str:
        parts = [
            chunk
            async for chunk in stream_grounded_answer_events(_grounded_answer())
        ]
        return "".join(parts)

    body = asyncio.run(collect())
    events = _parse_events(body)
    event_types = [event["type"] for event in events]

    assert event_types[0] == "start"
    assert "data-progress" in event_types
    assert "text-start" in event_types
    assert "text-delta" in event_types
    assert event_types[-1] == "finish"


def test_format_progress_event() -> None:
    from app.chat.streaming import format_progress_event

    event = format_progress_event("Searching indexed filings...")
    assert '"type":"data-progress"' in event.replace(" ", "")
    assert "Searching indexed filings..." in event


def test_stream_grounded_answer_emits_text_and_data_parts() -> None:
    async def collect() -> str:
        response = stream_grounded_answer(_grounded_answer())
        parts = [chunk async for chunk in response.body_iterator]
        return "".join(part.decode() if isinstance(part, bytes) else part for part in parts)

    body = asyncio.run(collect())
    events = _parse_events(body)
    event_types = [event["type"] for event in events]

    assert event_types[0] == "start"
    assert "text-start" in event_types
    assert "text-delta" in event_types
    assert "text-end" in event_types
    assert "data-citation" in event_types
    assert "data-source-passage" in event_types
    assert event_types[-1] == "finish"
    assert "data: [DONE]" in body

    citation_event = next(event for event in events if event["type"] == "data-citation")
    assert citation_event["data"]["citation_index"] == 1
    assert citation_event["data"]["chunk_id"] == str(CHUNK_ID)

    passage_event = next(event for event in events if event["type"] == "data-source-passage")
    assert passage_event["data"]["ticker"] == "AMZN"
    assert passage_event["data"]["content"] == "AWS operating income increased."

    text = "".join(event["delta"] for event in events if event["type"] == "text-delta")
    assert text == "AWS operating income rose [1]."

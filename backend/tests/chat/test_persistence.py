from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import patch
from uuid import UUID

from app.assistant.outputs import Citation, GroundedAnswer
from app.chat.persistence import assistant_answer_to_wire, enrich_assistant_messages
from app.database.chats import ChatMessageRecord, MessageCitationRecord
from app.retrieval.types import SourcePassage

MESSAGE_ID = UUID("880e8400-e29b-41d4-a716-446655440003")
THREAD_ID = UUID("770e8400-e29b-41d4-a716-446655440002")
CHUNK_ID = UUID("11111111-1111-1111-1111-111111111111")
DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


def _passage() -> SourcePassage:
    return SourcePassage(
        chunk_id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        chunk_index=3,
        content="Net income increased.",
        section="Item 8",
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0000320193-24-000123",
        filing_date=date(2024, 11, 1),
        source_url="https://example.com/aapl-10k",
        score=0.0,
    )


def test_assistant_answer_to_wire_includes_citations_and_passages() -> None:
    wire = assistant_answer_to_wire(
        GroundedAnswer(
            answer="Apple net income rose [1].",
            citations=[
                Citation(
                    citation_index=1,
                    chunk_id=CHUNK_ID,
                    excerpt="Net income increased.",
                )
            ],
            cited_passages=[_passage()],
        ),
        message_id=str(MESSAGE_ID),
    )

    part_types = [part["type"] for part in wire["parts"]]
    assert part_types == ["text", "data-citation", "data-source-passage"]
    assert wire["id"] == str(MESSAGE_ID)


def test_enrich_assistant_messages_rebuilds_message_data_from_citations() -> None:
    message = ChatMessageRecord(
        id=MESSAGE_ID,
        thread_id=THREAD_ID,
        role="assistant",
        content="Apple net income rose [1].",
        message_data=None,
        created_at=NOW,
    )
    citations = {
        MESSAGE_ID: [
            MessageCitationRecord(
                id=UUID("99999999-9999-9999-9999-999999999999"),
                message_id=MESSAGE_ID,
                chunk_id=CHUNK_ID,
                citation_index=1,
                excerpt="Net income increased.",
                created_at=NOW,
            )
        ]
    }

    with patch("app.chat.persistence._load_passages_by_chunk", return_value={CHUNK_ID: _passage()}):
        enriched = enrich_assistant_messages([message], citations)

    assert enriched[0].message_data is not None
    part_types = [part["type"] for part in enriched[0].message_data["parts"]]
    assert "data-citation" in part_types
    assert "data-source-passage" in part_types

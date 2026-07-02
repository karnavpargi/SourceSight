from datetime import date
from uuid import UUID

import pytest

from app.assistant.outputs import Citation, GroundedAnswer
from app.chat.messages import (
    ChatUIMessage,
    CitationData,
    CitationPart,
    SourcePassagePart,
    TextPart,
    extract_latest_user_text,
    grounded_answer_to_ui_message,
    message_text,
    parse_ui_message,
    ui_message_to_wire,
)
from app.retrieval.types import SourcePassage

CHUNK_ID = UUID("11111111-1111-1111-1111-111111111111")
DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")
MESSAGE_ID = "msg_abc123"


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


def test_extract_latest_user_text_from_parts() -> None:
    text = extract_latest_user_text(
        [
            {"id": "msg_1", "role": "assistant", "parts": [{"type": "text", "text": "Earlier answer"}]},
            {
                "id": "msg_2",
                "role": "user",
                "parts": [{"type": "text", "text": "AWS operating income"}],
            },
        ]
    )
    assert text == "AWS operating income"


def test_extract_latest_user_text_from_content_field() -> None:
    text = extract_latest_user_text(
        [{"id": "msg_1", "role": "user", "content": "  iPhone revenue  "}]
    )
    assert text == "iPhone revenue"


def test_extract_latest_user_text_raises_when_missing_user_message() -> None:
    with pytest.raises(ValueError, match="No user message"):
        extract_latest_user_text(
            [{"id": "msg_1", "role": "assistant", "content": "No question here"}]
        )


def test_parse_ui_message_round_trip() -> None:
    wire = {
        "id": MESSAGE_ID,
        "role": "assistant",
        "parts": [
            {"type": "text", "text": "AWS operating income rose [1]."},
            {
                "type": "data-citation",
                "data": {
                    "citation_index": 1,
                    "chunk_id": str(CHUNK_ID),
                    "excerpt": "AWS operating income increased.",
                },
            },
            {
                "type": "data-source-passage",
                "data": _passage().model_dump(mode="json"),
            },
        ],
    }

    message = parse_ui_message(wire)

    assert message == ChatUIMessage(
        id=MESSAGE_ID,
        role="assistant",
        parts=[
            TextPart(text="AWS operating income rose [1]."),
            CitationPart(
                data=CitationData(
                    citation_index=1,
                    chunk_id=CHUNK_ID,
                    excerpt="AWS operating income increased.",
                )
            ),
            SourcePassagePart(data=_passage()),
        ],
    )
    assert parse_ui_message(ui_message_to_wire(message)) == message


def test_grounded_answer_to_ui_message_builds_parts() -> None:
    answer = GroundedAnswer(
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

    message = grounded_answer_to_ui_message(answer, message_id=MESSAGE_ID)

    assert message.id == MESSAGE_ID
    assert message.role == "assistant"
    assert message_text(message) == "AWS operating income rose [1]."
    assert isinstance(message.parts[1], CitationPart)
    assert isinstance(message.parts[2], SourcePassagePart)


def test_parse_ui_message_rejects_unknown_part_type() -> None:
    with pytest.raises(ValueError, match="Unsupported message part type"):
        parse_ui_message(
            {
                "id": MESSAGE_ID,
                "role": "assistant",
                "parts": [{"type": "image", "url": "https://example.com/x.png"}],
            }
        )

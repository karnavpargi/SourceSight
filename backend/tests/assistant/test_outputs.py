from datetime import date
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.assistant.outputs import Citation, DraftCitation, GroundedAnswer, GroundedDraft, SourcePassage

CHUNK_ID = UUID("11111111-1111-1111-1111-111111111111")
DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")


def _sample_passage() -> SourcePassage:
    return SourcePassage(
        chunk_id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        chunk_index=3,
        content="AWS operating income increased.",
        section="Item 7. MD&A",
        page=None,
        ticker="AMZN",
        company_name="Amazon.com, Inc.",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001018724-24-000123",
        filing_date=date(2024, 2, 2),
        report_date=date(2023, 12, 31),
        source_url="https://example.com/amzn-10k",
        score=0.42,
        is_neighbor=False,
    )


def test_grounded_answer_round_trip() -> None:
    answer = GroundedAnswer(
        answer="AWS operating income rose year over year [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=CHUNK_ID,
                excerpt="AWS operating income increased.",
            )
        ],
        cited_passages=[_sample_passage()],
    )

    restored = GroundedAnswer.model_validate(answer.model_dump(mode="json"))

    assert restored == answer


def test_citation_rejects_empty_excerpt() -> None:
    with pytest.raises(ValidationError):
        Citation(citation_index=1, chunk_id=CHUNK_ID, excerpt="")


def test_citation_rejects_non_positive_index() -> None:
    with pytest.raises(ValidationError):
        Citation(citation_index=0, chunk_id=CHUNK_ID, excerpt="Evidence.")


def test_grounded_answer_rejects_duplicate_citation_indices() -> None:
    with pytest.raises(ValidationError, match="citation_index values must be unique"):
        GroundedAnswer(
            answer="Duplicate markers [1] and [1].",
            citations=[
                Citation(citation_index=1, chunk_id=CHUNK_ID, excerpt="First."),
                Citation(citation_index=1, chunk_id=CHUNK_ID, excerpt="Second."),
            ],
        )


def test_grounded_answer_allows_empty_citations() -> None:
    answer = GroundedAnswer(
        answer="This corpus does not contain enough evidence to answer that.",
        citations=[],
        cited_passages=[],
    )

    assert answer.citations == []
    assert answer.cited_passages == []


def test_grounded_draft_has_no_cited_passages_field() -> None:
    fields = GroundedDraft.model_fields
    assert "cited_passages" not in fields
    assert "chunk_id" not in DraftCitation.model_fields
    assert "evidence_alias" in DraftCitation.model_fields

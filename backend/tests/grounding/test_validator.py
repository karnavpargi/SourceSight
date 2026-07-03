from datetime import date
from uuid import UUID

import pytest

from app.assistant.outputs import Citation, GroundedAnswer
from app.grounding.validator import GroundingError, validate
from app.retrieval.types import SourcePassage

CHUNK_ID = UUID("11111111-1111-1111-1111-111111111111")
OTHER_CHUNK_ID = UUID("33333333-3333-3333-3333-333333333333")
DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")


def _passage(chunk_id: UUID = CHUNK_ID) -> SourcePassage:
    return SourcePassage(
        chunk_id=chunk_id,
        document_id=DOCUMENT_ID,
        chunk_index=3,
        content="AWS operating income increased.",
        ticker="AMZN",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001018724-24-000123",
        filing_date=date(2024, 2, 2),
        source_url="https://example.com/amzn-10k",
        score=0.42,
    )


def test_validate_allows_refusal_without_citations() -> None:
    answer = GroundedAnswer(answer="This corpus does not contain enough evidence to answer that.")
    validate(answer, [])


def test_validate_allows_refusal_with_contraction() -> None:
    answer = GroundedAnswer(
        answer="This corpus doesn't contain enough evidence to answer that."
    )
    validate(answer, [])


def test_validate_rejects_refusal_with_citations() -> None:
    answer = GroundedAnswer(
        answer="This corpus does not contain enough evidence to answer that.",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=CHUNK_ID,
                excerpt="AWS operating income increased.",
            )
        ],
    )
    with pytest.raises(GroundingError, match="must not include citations"):
        validate(answer, [_passage()])


def test_validate_allows_citations_to_retrieved_chunks() -> None:
    answer = GroundedAnswer(
        answer="AWS operating income rose [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=CHUNK_ID,
                excerpt="AWS operating income increased.",
            )
        ],
    )
    validate(answer, [_passage()])


def test_validate_rejects_grounded_answer_without_citations() -> None:
    answer = GroundedAnswer(answer="AWS operating income rose sharply.")
    with pytest.raises(GroundingError, match="must include at least one citation"):
        validate(answer, [_passage()])


def test_validate_rejects_unresolved_marker_in_answer() -> None:
    answer = GroundedAnswer(
        answer="AWS operating income rose [1] and margins improved [2].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=CHUNK_ID,
                excerpt="AWS operating income increased.",
            )
        ],
    )
    with pytest.raises(GroundingError, match="\\[2\\]"):
        validate(answer, [_passage()])


def test_validate_rejects_orphan_citation_record() -> None:
    answer = GroundedAnswer(
        answer="AWS operating income rose [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=CHUNK_ID,
                excerpt="AWS operating income increased.",
            ),
            Citation(
                citation_index=2,
                chunk_id=CHUNK_ID,
                excerpt="Margins improved.",
            ),
        ],
    )
    with pytest.raises(GroundingError, match="\\[2\\]"):
        validate(answer, [_passage()])


def test_validate_rejects_uncited_factual_segment() -> None:
    answer = GroundedAnswer(
        answer="AWS operating income rose sharply. Margins improved [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=CHUNK_ID,
                excerpt="Margins improved.",
            )
        ],
    )
    with pytest.raises(GroundingError, match="Uncited segment"):
        validate(answer, [_passage()])


def test_validate_rejects_citation_to_unretrieved_chunk() -> None:
    answer = GroundedAnswer(
        answer="AWS operating income rose [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=OTHER_CHUNK_ID,
                excerpt="Fabricated excerpt.",
            )
        ],
    )
    with pytest.raises(GroundingError, match="was not retrieved"):
        validate(answer, [_passage()])

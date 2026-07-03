from datetime import date
from uuid import UUID

from app.assistant.outputs import GroundedAnswer
from app.grounding.repair import repair_grounded_answer
from app.retrieval.types import SourcePassage

CHUNK_A = UUID("11111111-1111-1111-1111-111111111111")
CHUNK_B = UUID("33333333-3333-3333-3333-333333333333")
DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")


def _passage(chunk_id: UUID, content: str) -> SourcePassage:
    return SourcePassage(
        chunk_id=chunk_id,
        document_id=DOCUMENT_ID,
        chunk_index=1,
        content=content,
        ticker="AMZN",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001018724-24-000123",
        filing_date=date(2024, 2, 2),
        source_url="https://example.com/amzn-10k",
        score=0.42,
    )


def test_repair_builds_missing_citation_records_from_cited_passages() -> None:
    passages = [
        _passage(CHUNK_A, "AWS operating income increased."),
        _passage(CHUNK_B, "Segment margins expanded."),
    ]
    answer = GroundedAnswer(
        answer="AWS operating income rose [1] and margins expanded [2].",
        cited_passages=passages,
    )

    repaired = repair_grounded_answer(answer, passages)

    assert [citation.citation_index for citation in repaired.citations] == [1, 2]
    assert repaired.citations[0].chunk_id == CHUNK_A
    assert repaired.citations[1].chunk_id == CHUNK_B
    assert len(repaired.cited_passages) == 2


def test_repair_leaves_already_valid_answer_unchanged() -> None:
    from app.assistant.outputs import Citation

    passage = _passage(CHUNK_A, "AWS operating income increased.")
    answer = GroundedAnswer(
        answer="AWS operating income rose [1].",
        citations=[
            Citation(
                citation_index=1,
                chunk_id=CHUNK_A,
                excerpt="AWS operating income increased.",
            )
        ],
        cited_passages=[passage],
    )

    repaired = repair_grounded_answer(answer, [passage])

    assert repaired == answer

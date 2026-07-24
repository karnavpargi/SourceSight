from datetime import date
from uuid import uuid4

from app.assistant.evidence import EvidenceRegistry
from app.assistant.finalize import finalize_grounded_draft
from app.assistant.outputs import DraftCitation, GroundedDraft
from app.retrieval.types import SourcePassage


def test_finalize_maps_aliases_to_uuid_citations_and_passages() -> None:
    passage = SourcePassage(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        content="AWS operating income was $39,834 million.",
        section="Item 8",
        page=None,
        ticker="AMZN",
        company_name="Amazon",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001",
        filing_date=date(2025, 1, 31),
        report_date=None,
        source_url="https://example.com",
        score=1.0,
    )
    registry = EvidenceRegistry()
    registry.register([passage])
    draft = GroundedDraft(
        answer="AWS operating income was $39,834 million [1].",
        citations=[
            DraftCitation(
                citation_index=1,
                evidence_alias="E1",
                excerpt="AWS operating income was $39,834 million.",
            )
        ],
    )
    answer = finalize_grounded_draft(draft, registry)
    assert answer.citations[0].chunk_id == passage.chunk_id
    assert answer.cited_passages == [passage]
    assert answer.answer == draft.answer


def test_finalize_rejects_unknown_alias() -> None:
    draft = GroundedDraft(
        answer="Claim [1].",
        citations=[DraftCitation(citation_index=1, evidence_alias="E9", excerpt="x")],
    )
    try:
        finalize_grounded_draft(draft, EvidenceRegistry())
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "E9" in str(exc)

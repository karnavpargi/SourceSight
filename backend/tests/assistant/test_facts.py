from datetime import date
from uuid import uuid4

import pytest

from app.assistant.evidence import EvidenceRegistry
from app.retrieval.types import SourcePassage
from app.assistant.outputs import GroundedDraft, DraftCitation

from app.assistant.facts import (
    FactExtraction,
    ExtractedFact,
    validate_draft_numeric_claims,
    validate_extraction,
)


def _registry(content: str) -> EvidenceRegistry:
    registry = EvidenceRegistry()
    registry.register(
        [
            SourcePassage(
                chunk_id=uuid4(),
                document_id=uuid4(),
                chunk_index=0,
                content=content,
                section="Item 8",
                ticker="AMZN",
                company_name="Amazon",
                form_type="10-K",
                fiscal_year=2024,
                accession_number="0001",
                filing_date=date(2025, 1, 31),
                source_url="https://example.com",
                score=1.0,
            )
        ]
    )
    return registry


def _draft(answer: str, alias: str) -> GroundedDraft:
    return GroundedDraft(
        answer=answer,
        citations=[
            DraftCitation(
                citation_index=1,
                evidence_alias=alias,
                excerpt="Revenue was disclosed in the segment table.",
            )
        ],
    )


def test_supported_numeric_fact_requires_known_alias_and_source_value() -> None:
    registry = _registry("Revenue was $39,834 million.")
    extraction = FactExtraction(
        facts=[
            ExtractedFact(
                status="supported",
                ticker="AMZN",
                fiscal_year=2024,
                topic="AWS operating income",
                value="$39,834 million",
                unit="USD millions",
                finding=None,
                evidence_alias="E1",
            )
        ],
        missing_scope=[],
        conflicts=[],
        draft=_draft("AWS operating income was $39,834 million [1].", "E1"),
    )
    validated = validate_extraction(extraction, registry, route="extractive")
    assert len(validated.facts) == 1


def test_numeric_fact_not_present_in_source_is_discarded() -> None:
    registry = _registry("Revenue was $10 million.")
    extraction = FactExtraction(
        facts=[
            ExtractedFact(
                status="supported",
                ticker="AMZN",
                fiscal_year=2024,
                topic="AWS operating income",
                value="$99 million",
                unit="USD millions",
                finding=None,
                evidence_alias="E1",
            )
        ],
        missing_scope=[],
        conflicts=[],
        draft=_draft("AWS operating income was $99 million [1].", "E1"),
    )
    validated = validate_extraction(extraction, registry, route="extractive")
    assert validated.facts == []
    assert "unsupported numeric value" in validated.validation_errors[0]


def test_numeric_fact_requires_all_source_numbers() -> None:
    registry = _registry("Revenue was $10 million.")
    extraction = FactExtraction(
        facts=[
            ExtractedFact(
                status="supported",
                ticker="AMZN",
                fiscal_year=2024,
                topic="revenue change",
                value="$10 million, up 25%",
                unit="USD millions",
                finding=None,
                evidence_alias="E1",
            )
        ],
        missing_scope=[],
        conflicts=[],
        draft=_draft("Revenue was $10 million, up 25% [1].", "E1"),
    )

    validated = validate_extraction(extraction, registry, route="extractive")

    assert validated.facts == []
    assert "unsupported numeric value" in validated.validation_errors[0]


def test_draft_numeric_claim_requires_all_numbers_in_cited_passage() -> None:
    registry = _registry("Revenue was $10 million.")
    draft = _draft("Revenue was $10 million, up 25% [1].", "E1")

    with pytest.raises(ValueError, match="unsupported numeric claim"):
        validate_draft_numeric_claims(draft, registry)


def test_draft_numeric_claim_allows_supported_multiple_numbers() -> None:
    registry = _registry("Revenue was $39,834 million, up 13.2%.")
    draft = _draft("Revenue was $39,834 million, up 13.2% [1].", "E1")

    validate_draft_numeric_claims(draft, registry)


def test_synthesis_route_rejects_extractive_draft() -> None:
    extraction = FactExtraction(
        facts=[
            ExtractedFact(
                status="supported",
                ticker="AMZN",
                fiscal_year=2024,
                topic="demand",
                value=None,
                unit=None,
                finding="Demand increased.",
                evidence_alias="E1",
            )
        ],
        missing_scope=[],
        conflicts=[],
        draft=_draft("Demand increased [1].", "E1"),
    )
    with pytest.raises(ValueError, match="must omit draft"):
        validate_extraction(
            extraction,
            _registry("Demand increased."),
            route="synthesis",
        )


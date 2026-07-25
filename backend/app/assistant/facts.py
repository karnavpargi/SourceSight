from __future__ import annotations

from dataclasses import dataclass
import re
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, model_validator

from app.assistant.outputs import GroundedDraft
from app.assistant.evidence import EvidenceRegistry

__all__ = [
    "FactStatus",
    "ExtractedFact",
    "FactExtraction",
    "ValidatedExtraction",
    "validate_draft_numeric_claims",
    "validate_extraction",
]

FactStatus = Literal["supported", "missing", "conflicting"]


class ExtractedFact(BaseModel):
    status: FactStatus
    ticker: str | None = None
    fiscal_year: int | None = None
    topic: str
    value: str | None = None
    unit: str | None = None
    finding: str | None = None
    evidence_alias: str | None = None

    @model_validator(mode="after")
    def support_has_evidence(self) -> "ExtractedFact":
        if self.status in {"supported", "conflicting"} and not self.evidence_alias:
            raise ValueError("supported/conflicting facts require evidence_alias")
        if self.status == "missing" and self.evidence_alias is not None:
            raise ValueError("missing facts cannot cite evidence")
        return self


class FactExtraction(BaseModel):
    facts: list[ExtractedFact]
    missing_scope: list[str]
    conflicts: list[str]
    draft: GroundedDraft | None = None


@dataclass(frozen=True)
class ValidatedExtraction:
    facts: list[ExtractedFact]
    missing_scope: tuple[str, ...]
    conflicts: tuple[str, ...]
    draft: GroundedDraft | None
    validation_errors: tuple[str, ...]


_NUM_RE = re.compile(r'[\(\)\-\$€£]?\d[\d,\.]*[\)]?')
_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_YEAR_REFERENCE_RE = re.compile(
    r"\bFY\s*(?:19|20)\d{2}\b"
    r"|\bfiscal\s+(?:year\s+)?(?:19|20)\d{2}\b"
    r"|\b(?:19|20)\d{2}\s*[–-]\s*(?:19|20)\d{2}\b"
    r"|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_SUBSTANTIVE_NUMERIC_RE = re.compile(
    r"[$%]"
    r"|\$\s*\d+(?:\.\d+)?"
    r"|\b\d+(?:\.\d+)?(?:%|\s*(?:million|billion|trillion|bn|m|b)\b)",
    re.IGNORECASE,
)


def _numeric_tokens(text: str) -> set[Decimal]:
    """Extract numeric tokens from text as Decimals."""
    found: set[Decimal] = set()
    for m in _NUM_RE.finditer(text or ""):
        raw = m.group(0)
        # remove currency symbols
        s = re.sub(r'[\$€£]', '', raw)
        s = s.strip()
        negative = False
        if s.startswith('(') and s.endswith(')'):
            negative = True
            s = s[1:-1]
        if s.startswith('-'):
            negative = True
            s = s[1:]
        s = s.replace(',', '')
        if s == "":
            continue
        try:
            d = Decimal(s)
        except InvalidOperation:
            continue
        if negative:
            d = -d
        found.add(d)
    return found


def _validate_alias_exists(alias: str, registry: EvidenceRegistry) -> bool:
    try:
        registry.resolve(alias)
        return True
    except KeyError:
        return False


def validate_draft_numeric_claims(
    draft: GroundedDraft,
    evidence: EvidenceRegistry,
) -> None:
    """Reject cited numeric claims whose values are absent from their evidence."""
    aliases_by_index = {
        citation.citation_index: citation.evidence_alias
        for citation in draft.citations
    }
    for paragraph in draft.answer.splitlines():
        for segment in _SENTENCE_SPLIT_RE.split(paragraph.strip()):
            indexes = {
                int(index)
                for index in _CITATION_MARKER_RE.findall(segment)
                if int(index) in aliases_by_index
            }
            if not indexes:
                continue
            claim = _CITATION_MARKER_RE.sub("", segment)
            claim_without_years = _YEAR_REFERENCE_RE.sub("", claim)
            if not _SUBSTANTIVE_NUMERIC_RE.search(claim_without_years):
                continue
            claim_numbers = _numeric_tokens(claim_without_years)
            if not claim_numbers:
                continue
            for index in indexes:
                passage = evidence.resolve(aliases_by_index[index])
                if not claim_numbers <= _numeric_tokens(passage.content):
                    raise ValueError("unsupported numeric claim in cited draft")


def validate_extraction(extraction: FactExtraction, evidence: EvidenceRegistry, route: str) -> ValidatedExtraction:
    errors: list[str] = []
    validated_facts: list[ExtractedFact] = []

    # route / draft requirements
    if route == "extractive":
        if extraction.draft is None:
            raise ValueError("extractive route requires draft")
    elif route in {"synthesis", "boundary"}:
        if extraction.draft is not None:
            raise ValueError("must omit draft")

    # validate draft citations aliases
    if extraction.draft:
        for c in extraction.draft.citations:
            if not _validate_alias_exists(c.evidence_alias, evidence):
                errors.append(f"unknown evidence alias in draft: {c.evidence_alias}")

    for fact in extraction.facts:
        # validate alias existence for facts that cite evidence
        if fact.evidence_alias:
            try:
                passage = evidence.resolve(fact.evidence_alias)
            except KeyError:
                errors.append(f"unknown evidence alias: {fact.evidence_alias}")
                # skip fact if alias unknown
                continue
        else:
            passage = None

        # numeric validation: if fact has a value containing numeric tokens and status supported,
        # ensure those tokens appear in the cited passage content.
        if fact.status == "supported" and fact.value and fact.evidence_alias:
            fact_nums = _numeric_tokens(fact.value)
            src_nums = _numeric_tokens(passage.content if passage is not None else "")
            if fact_nums and not fact_nums <= src_nums:
                errors.append("unsupported numeric value")
                continue

        # passed all checks
        validated_facts.append(fact)

    return ValidatedExtraction(
        facts=validated_facts,
        missing_scope=tuple(extraction.missing_scope),
        conflicts=tuple(extraction.conflicts),
        draft=extraction.draft,
        validation_errors=tuple(errors),
    )


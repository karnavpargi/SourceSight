from __future__ import annotations

from app.assistant.evidence import EvidenceRegistry
from app.assistant.outputs import Citation, GroundedAnswer, GroundedDraft


def finalize_grounded_draft(
    draft: GroundedDraft,
    registry: EvidenceRegistry,
) -> GroundedAnswer:
    citations: list[Citation] = []
    cited = []
    seen = set()
    for item in draft.citations:
        try:
            passage = registry.resolve(item.evidence_alias)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        citations.append(
            Citation(
                citation_index=item.citation_index,
                chunk_id=passage.chunk_id,
                excerpt=item.excerpt,
            )
        )
        if passage.chunk_id not in seen:
            cited.append(passage)
            seen.add(passage.chunk_id)
    return GroundedAnswer(
        answer=draft.answer,
        citations=citations,
        cited_passages=cited,
    )

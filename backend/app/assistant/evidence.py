from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from pydantic import BaseModel, Field

from app.retrieval.types import SourcePassage

__all__ = ["CompactEvidence", "EvidenceRegistry"]

_MAX_COMPACT_CONTENT_CHARS = 1200


class CompactEvidence(BaseModel):
    alias: str = Field(description="Turn-local evidence id, e.g. E1.")
    content: str
    ticker: str
    fiscal_year: int
    section: str | None = None


def _truncate_content(content: str, *, max_length: int = _MAX_COMPACT_CONTENT_CHARS) -> str:
    """Return a compact content string capped at max_length characters.

    The full SourcePassage content is retained in the registry; this only
    affects what is sent back to the model in CompactEvidence.
    """
    if len(content) <= max_length:
        return content
    # Reserve one character for the ellipsis so the final string length
    # does not exceed max_length.
    return content[: max_length - 1] + "…"


@dataclass
class EvidenceRegistry:
    max_passages: int = 8
    _by_alias: dict[str, SourcePassage] = field(default_factory=dict)
    _alias_by_chunk: dict[UUID, str] = field(default_factory=dict)

    def register(self, passages: list[SourcePassage]) -> list[CompactEvidence]:
        emitted: list[CompactEvidence] = []
        for passage in passages:
            if passage.chunk_id in self._alias_by_chunk:
                continue
            if len(self._by_alias) >= self.max_passages:
                break
            alias = f"E{len(self._by_alias) + 1}"
            self._by_alias[alias] = passage
            self._alias_by_chunk[passage.chunk_id] = alias
            emitted.append(self._to_compact(alias, passage))
        return emitted

    def resolve(self, alias: str) -> SourcePassage:
        try:
            return self._by_alias[alias]
        except KeyError as exc:
            raise KeyError(f"Unknown evidence alias: {alias}") from exc

    def all_passages(self) -> list[SourcePassage]:
        return list(self._by_alias.values())

    def compact_dump(self) -> list[CompactEvidence]:
        """Return compact rows for all registered aliases.

        The compact rows use truncated content suitable for passing back to the model.
        """
        return [self._to_compact(alias, passage) for alias, passage in self._by_alias.items()]

    def _to_compact(self, alias: str, passage: SourcePassage) -> CompactEvidence:
        return CompactEvidence(
            alias=alias,
            content=_truncate_content(passage.content),
            ticker=passage.ticker,
            fiscal_year=passage.fiscal_year,
            section=passage.section,
        )

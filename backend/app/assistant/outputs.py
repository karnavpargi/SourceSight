"""Typed output models for the document assistant agent."""

from __future__ import annotations

from typing import Self
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.retrieval.types import SourcePassage

__all__ = ["Citation", "GroundedAnswer", "SourcePassage"]


class Citation(BaseModel):
    """A single in-answer citation pointing at a retrieved chunk."""

    citation_index: int = Field(
        ge=1,
        description="1-based marker index shown in the answer text, e.g. [1].",
    )
    chunk_id: UUID = Field(description="Retrieved document chunk backing this citation.")
    excerpt: str = Field(
        min_length=1,
        description="Quoted passage text supporting the cited claim.",
    )


class GroundedAnswer(BaseModel):
    """Structured assistant output: prose answer plus citations and source passages."""

    answer: str = Field(
        min_length=1,
        description="Analyst-facing answer text with inline citation markers.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Normalized citation records referenced by the answer.",
    )
    cited_passages: list[SourcePassage] = Field(
        default_factory=list,
        description="Retrieved passages cited in the answer, with filing metadata.",
    )

    @model_validator(mode="after")
    def citation_indices_are_unique(self) -> Self:
        indices = [citation.citation_index for citation in self.citations]
        if len(indices) != len(set(indices)):
            raise ValueError("citation_index values must be unique within an answer")
        return self

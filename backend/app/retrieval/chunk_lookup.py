"""Chunk lookup helpers for agent tool calls."""

from __future__ import annotations

from uuid import UUID

from pydantic_ai.exceptions import ModelRetry


def chunk_not_found_retry(chunk_id: UUID) -> ModelRetry:
    return ModelRetry(
        f"Chunk {chunk_id} was not found. Only use chunk_id values returned from "
        "search_filings in this turn. Prefer the passage text already returned by "
        "search_filings instead of calling read_chunk again."
    )

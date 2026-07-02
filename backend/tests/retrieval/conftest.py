"""Fixtures for retrieval integration tests."""

from __future__ import annotations

import httpx
import pytest
from sqlalchemy import text

from app.config import settings
from ingest.db import get_engine

MIN_CHUNK_COUNT = 1000


@pytest.fixture(scope="module")
def ingested_corpus() -> None:
    """Skip unless the local SEC corpus has been ingested into Postgres."""
    try:
        with get_engine().connect() as conn:
            chunk_count = conn.execute(text("select count(*) from document_chunks")).scalar_one()
    except Exception as exc:
        pytest.skip(f"Database unavailable: {exc}")

    if chunk_count < MIN_CHUNK_COUNT:
        pytest.skip(
            f"Expected ingested corpus (>{MIN_CHUNK_COUNT} chunks), found {chunk_count}. "
            "Run `uv run python -m ingest.run` first."
        )


@pytest.fixture(scope="module")
def ollama_embeddings() -> None:
    """Skip unless Ollama is reachable for query embeddings."""
    if settings.embedding_provider != "ollama":
        pytest.skip(f"Integration retrieval tests require EMBEDDING_PROVIDER=ollama, got {settings.embedding_provider!r}.")

    try:
        response = httpx.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=3.0)
        response.raise_for_status()
    except Exception as exc:
        pytest.skip(f"Ollama unavailable at {settings.ollama_base_url}: {exc}")

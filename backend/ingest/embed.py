"""Embedding via local Ollama."""

from __future__ import annotations

from app.config import settings
from ingest.providers.ollama import embed_texts_ollama


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using the local Ollama model configured in settings."""
    embeddings = embed_texts_ollama(texts)
    _validate_dimensions(embeddings)
    return embeddings


def _validate_dimensions(embeddings: list[list[float]]) -> None:
    if not embeddings:
        return
    actual = len(embeddings[0])
    expected = settings.embedding_dimensions
    if actual != expected:
        raise ValueError(
            f"Ollama returned {actual}-dim vectors but EMBEDDING_DIMENSIONS={expected}. "
            "Update the model or migration to match."
        )

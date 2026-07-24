"""Embedding provider routing for ingestion and query-time retrieval."""

from __future__ import annotations

from app.config import settings
from ingest.providers.google import EmbeddingTaskType, embed_texts_google
from ingest.providers.ollama import embed_texts_ollama


def embed_texts(
    texts: list[str],
    *,
    task_type: EmbeddingTaskType = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    if not texts:
        return []

    if settings.embedding_provider == "none":
        return []

    if settings.embedding_provider == "ollama":
        vectors = embed_texts_ollama(texts)
    else:
        vectors = embed_texts_google(texts, task_type=task_type)

    expected = settings.embedding_dimensions
    for index, vector in enumerate(vectors):
        if len(vector) != expected:
            raise ValueError(
                f"embedding provider {settings.embedding_provider!r} returned "
                f"{len(vector)} dimensions for text {index}; expected {expected}"
            )
    return vectors


def embed_query(query: str) -> list[float]:
    vectors = embed_texts([query], task_type="RETRIEVAL_QUERY")
    if not vectors:
        raise RuntimeError("embedding provider is disabled (EMBEDDING_PROVIDER=none)")
    return vectors[0]

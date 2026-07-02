"""Embedding dispatcher — routes to the configured provider."""

from __future__ import annotations

from collections.abc import Callable

from app.config import settings
from ingest.providers.ollama import embed_texts_ollama
from ingest.providers.openai import embed_texts_openai

ProviderFn = Callable[[list[str]], list[list[float]]]

_PROVIDERS: dict[str, ProviderFn] = {
    "openai": lambda texts: embed_texts_openai(texts),
    "ollama": lambda texts: embed_texts_ollama(texts),
}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using the provider selected in settings."""
    provider = settings.embedding_provider.lower()
    embed_fn = _PROVIDERS.get(provider)
    if embed_fn is None:
        supported = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER={provider!r}. Supported: {supported}")

    embeddings = embed_fn(texts)
    _validate_dimensions(embeddings, provider)
    return embeddings


def _validate_dimensions(embeddings: list[list[float]], provider: str) -> None:
    if not embeddings:
        return
    actual = len(embeddings[0])
    expected = settings.embedding_dimensions
    if actual != expected:
        raise ValueError(
            f"{provider} returned {actual}-dim vectors but EMBEDDING_DIMENSIONS={expected}. "
            "Update the model or migration to match."
        )

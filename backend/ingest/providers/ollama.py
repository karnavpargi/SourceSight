"""Ollama embedding provider."""

from __future__ import annotations

import time

import httpx

from app.config import settings
from app.http_client import get_sync_client

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0


def embed_texts_ollama(
    texts: list[str],
    *,
    batch_size: int = 32,
    client: httpx.Client | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    embeddings: list[list[float]] = []
    owns_client = client is None
    http_client = client or get_sync_client()

    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            embeddings.extend(_embed_batch_with_retry(http_client, batch))
    finally:
        if owns_client and http_client is not get_sync_client():
            http_client.close()

    return embeddings


def _embed_batch_with_retry(client: httpx.Client, texts: list[str]) -> list[list[float]]:
    backoff = INITIAL_BACKOFF_SECONDS
    url = f"{settings.ollama_base_url.rstrip('/')}/api/embed"
    payload = {
        "model": settings.ollama_embedding_model,
        "input": texts,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = client.post(url, json=payload)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == MAX_RETRIES - 1:
                    response.raise_for_status()
                time.sleep(backoff)
                backoff *= 2
                continue

            response.raise_for_status()
            data = response.json()
            batch_embeddings = data.get("embeddings")
            if not isinstance(batch_embeddings, list):
                raise RuntimeError("Ollama /api/embed response missing embeddings list")
            return batch_embeddings
        except httpx.HTTPStatusError:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(backoff)
            backoff *= 2
        except httpx.RequestError:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError("ollama embedding retry loop exited unexpectedly")

"""OpenAI embedding provider."""

from __future__ import annotations

import time

from openai import APIStatusError, OpenAI, RateLimitError

from app.config import settings

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0


def embed_texts_openai(
    texts: list[str],
    *,
    batch_size: int = 100,
    client: OpenAI | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    openai_client = client or OpenAI(api_key=settings.openai_api_key)
    embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings.extend(_embed_batch_with_retry(openai_client, batch))

    return embeddings


def _embed_batch_with_retry(client: OpenAI, texts: list[str]) -> list[list[float]]:
    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model=settings.openai_embedding_model,
                input=texts,
                dimensions=settings.embedding_dimensions,
            )
            return [item.embedding for item in response.data]
        except (RateLimitError, APIStatusError) as exc:
            if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError("embedding retry loop exited unexpectedly")


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error", {})
            if error.get("code") == "insufficient_quota":
                return False
        return True
    if isinstance(exc, APIStatusError):
        if exc.status_code == 429:
            body = exc.body if isinstance(exc.body, dict) else {}
            error = body.get("error", {})
            if error.get("code") == "insufficient_quota":
                return False
        return exc.status_code == 429 or exc.status_code >= 500
    return False

"""Google Gemini embedding provider."""

from __future__ import annotations

import time
from typing import Literal

import httpx

from app.config import settings

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 1.0
GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

EmbeddingTaskType = Literal["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT"]

_NON_RETRYABLE_STATUS_CODES = {400, 401, 403}


def _google_api_key() -> str:
    api_key = settings.google_api_key.strip()
    if not api_key:
        raise RuntimeError("Google API key is missing (settings.google_api_key).")
    return api_key


def _google_api_headers(api_key: str) -> dict[str, str]:
    # AI Studio keys (including AQ.* keys) authenticate via header, not ?key=.
    return {"Content-Type": "application/json", "X-goog-api-key": api_key}


def _post_google_embeddings(
    client: httpx.Client, url: str, *, api_key: str, json: dict[str, object]
) -> httpx.Response:
    return client.post(url, headers=_google_api_headers(api_key), json=json)


def embed_texts_google(
    texts: list[str],
    *,
    task_type: EmbeddingTaskType = "RETRIEVAL_DOCUMENT",
    batch_size: int = 32,
    client: httpx.Client | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    owns_client = client is None
    http_client = client or httpx.Client(timeout=120.0)
    model = settings.google_embedding_model
    dimensions = settings.embedding_dimensions
    embeddings: list[list[float]] = []

    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            if len(batch) == 1:
                embeddings.append(
                    _embed_single_with_retry(
                        http_client,
                        batch[0],
                        model=model,
                        task_type=task_type,
                        dimensions=dimensions,
                    )
                )
            else:
                embeddings.extend(
                    _embed_batch_with_retry(
                        http_client,
                        batch,
                        model=model,
                        task_type=task_type,
                        dimensions=dimensions,
                    )
                )
    finally:
        if owns_client:
            http_client.close()

    return embeddings


def _embed_single_with_retry(
    client: httpx.Client,
    text: str,
    *,
    model: str,
    task_type: EmbeddingTaskType,
    dimensions: int,
) -> list[float]:
    url = f"{GOOGLE_API_BASE}/models/{model}:embedContent"
    payload = _embed_request_payload(text, model=model, task_type=task_type, dimensions=dimensions)
    backoff = INITIAL_BACKOFF_SECONDS
    api_key = _google_api_key()

    for attempt in range(MAX_RETRIES):
        try:
            response = _post_google_embeddings(client, url, api_key=api_key, json=payload)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == MAX_RETRIES - 1:
                    response.raise_for_status()
                time.sleep(backoff)
                backoff *= 2
                continue

            response.raise_for_status()
            values = response.json().get("embedding", {}).get("values")
            if not isinstance(values, list):
                raise RuntimeError("Google embedContent response missing embedding.values")
            return values
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in _NON_RETRYABLE_STATUS_CODES:
                raise
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(backoff)
            backoff *= 2
        except httpx.RequestError:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError("google embedding retry loop exited unexpectedly")


def _embed_batch_with_retry(
    client: httpx.Client,
    texts: list[str],
    *,
    model: str,
    task_type: EmbeddingTaskType,
    dimensions: int,
) -> list[list[float]]:
    url = f"{GOOGLE_API_BASE}/models/{model}:batchEmbedContents"
    payload = {
        "requests": [
            _embed_request_payload(text, model=model, task_type=task_type, dimensions=dimensions)
            for text in texts
        ]
    }
    backoff = INITIAL_BACKOFF_SECONDS
    api_key = _google_api_key()

    for attempt in range(MAX_RETRIES):
        try:
            response = _post_google_embeddings(client, url, api_key=api_key, json=payload)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == MAX_RETRIES - 1:
                    response.raise_for_status()
                time.sleep(backoff)
                backoff *= 2
                continue

            response.raise_for_status()
            entries = response.json().get("embeddings")
            if not isinstance(entries, list):
                raise RuntimeError("Google batchEmbedContents response missing embeddings list")

            vectors: list[list[float]] = []
            for entry in entries:
                values = entry.get("values")
                if values is None and isinstance(entry.get("embedding"), dict):
                    values = entry["embedding"].get("values")
                if not isinstance(values, list):
                    raise RuntimeError("Google batchEmbedContents entry missing embedding values")
                vectors.append(values)
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Google batchEmbedContents returned {len(vectors)} vectors for {len(texts)} texts"
                )
            return vectors
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in _NON_RETRYABLE_STATUS_CODES:
                raise
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(backoff)
            backoff *= 2
        except httpx.RequestError:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError("google embedding retry loop exited unexpectedly")


def _embed_request_payload(
    text: str,
    *,
    model: str,
    task_type: EmbeddingTaskType,
    dimensions: int,
) -> dict[str, object]:
    return {
        "model": f"models/{model}",
        "content": {"parts": [{"text": text}]},
        "taskType": task_type,
        "outputDimensionality": dimensions,
    }

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from ingest.providers import ollama


def test_ollama_request_error_on_final_attempt() -> None:
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.RequestError("network")
    with patch("ingest.providers.ollama.MAX_RETRIES", 1), pytest.raises(httpx.RequestError):
        ollama.embed_texts_ollama(["x"], client=mock_client)


def test_ollama_embed_empty() -> None:
    assert ollama.embed_texts_ollama([]) == []


def test_ollama_embed_success() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"embeddings": [[0.1], [0.2]]}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    result = ollama.embed_texts_ollama(["a", "b"], client=mock_client)
    assert result == [[0.1], [0.2]]
    mock_client.post.assert_called()


def test_ollama_retries_server_error() -> None:
    failing = MagicMock(status_code=503)
    failing.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=failing
    )
    success = MagicMock(status_code=200)
    success.json.return_value = {"embeddings": [[0.9]]}
    mock_client = MagicMock()
    mock_client.post.side_effect = [failing, success]

    with patch("ingest.providers.ollama.time.sleep"):
        result = ollama.embed_texts_ollama(["x"], client=mock_client)
    assert result == [[0.9]]


def test_ollama_retries_request_error() -> None:
    success = MagicMock(status_code=200)
    success.json.return_value = {"embeddings": [[0.8]]}
    mock_client = MagicMock()
    mock_client.post.side_effect = [httpx.RequestError("network"), success]

    with patch("ingest.providers.ollama.time.sleep"):
        result = ollama.embed_texts_ollama(["x"], client=mock_client)
    assert result == [[0.8]]


def test_ollama_missing_embeddings_raises() -> None:
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"model": "nomic-embed-text"}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with pytest.raises(RuntimeError, match="missing embeddings"):
        ollama.embed_texts_ollama(["x"], client=mock_client)


def test_ollama_closes_owned_client() -> None:
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"embeddings": [[0.1]]}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    with patch("ingest.providers.ollama.httpx.Client", return_value=mock_client):
        result = ollama.embed_texts_ollama(["x"])
    assert result == [[0.1]]
    mock_client.close.assert_called_once()


def test_ollama_http_status_error_on_final_attempt() -> None:
    failing = MagicMock(status_code=500)
    failing.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=failing
    )
    mock_client = MagicMock()
    mock_client.post.return_value = failing
    with patch("ingest.providers.ollama.MAX_RETRIES", 1), pytest.raises(httpx.HTTPStatusError):
        ollama.embed_texts_ollama(["x"], client=mock_client)


def test_ollama_http_status_error_retries_before_success() -> None:
    failing = MagicMock(status_code=404)
    failing.raise_for_status.side_effect = httpx.HTTPStatusError(
        "missing", request=MagicMock(), response=failing
    )
    success = MagicMock(status_code=200)
    success.json.return_value = {"embeddings": [[0.4]]}
    mock_client = MagicMock()
    mock_client.post.side_effect = [failing, success]

    with patch("ingest.providers.ollama.time.sleep"):
        result = ollama.embed_texts_ollama(["x"], client=mock_client)
    assert result == [[0.4]]


def test_ollama_retry_loop_exhaustion() -> None:
    mock_client = MagicMock()
    mock_client.post.return_value = MagicMock(status_code=429)
    with patch("ingest.providers.ollama.MAX_RETRIES", 0), pytest.raises(RuntimeError):
        ollama._embed_batch_with_retry(mock_client, ["x"])


def test_providers_init_exports() -> None:
    from ingest.providers import embed_texts_ollama

    assert callable(embed_texts_ollama)

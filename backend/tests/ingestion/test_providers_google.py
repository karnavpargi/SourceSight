from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from ingest.providers import google


def test_google_embed_empty() -> None:
    assert google.embed_texts_google([]) == []


def test_google_embed_single_success() -> None:
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"embedding": {"values": [0.1, 0.2]}}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    result = google.embed_texts_google(["hello"], client=mock_client)
    assert result == [[0.1, 0.2]]
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert "embedContent" in call_kwargs.args[0]
    assert call_kwargs.kwargs["json"]["taskType"] == "RETRIEVAL_DOCUMENT"


def test_google_embed_batch_success() -> None:
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {
        "embeddings": [{"values": [0.1]}, {"values": [0.2]}],
    }
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    result = google.embed_texts_google(["a", "b"], client=mock_client)
    assert result == [[0.1], [0.2]]
    assert "batchEmbedContents" in mock_client.post.call_args.args[0]


def test_google_embed_query_task_type() -> None:
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"embedding": {"values": [0.5]}}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    google.embed_texts_google(["query"], task_type="RETRIEVAL_QUERY", client=mock_client)
    payload = mock_client.post.call_args.kwargs["json"]
    assert payload["taskType"] == "RETRIEVAL_QUERY"


def test_google_retries_server_error() -> None:
    failing = MagicMock(status_code=503)
    failing.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=failing
    )
    success = MagicMock(status_code=200)
    success.json.return_value = {"embedding": {"values": [0.9]}}
    mock_client = MagicMock()
    mock_client.post.side_effect = [failing, success]

    with patch("ingest.providers.google.time.sleep"):
        result = google.embed_texts_google(["x"], client=mock_client)
    assert result == [[0.9]]


def test_google_retries_request_error() -> None:
    success = MagicMock(status_code=200)
    success.json.return_value = {"embedding": {"values": [0.8]}}
    mock_client = MagicMock()
    mock_client.post.side_effect = [httpx.RequestError("network"), success]

    with patch("ingest.providers.google.time.sleep"):
        result = google.embed_texts_google(["x"], client=mock_client)
    assert result == [[0.8]]


def test_google_missing_single_embedding_raises() -> None:
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"model": "gemini-embedding-001"}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with pytest.raises(RuntimeError, match="missing embedding.values"):
        google.embed_texts_google(["x"], client=mock_client)


def test_google_missing_batch_embeddings_raises() -> None:
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"model": "gemini-embedding-001"}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with pytest.raises(RuntimeError, match="missing embeddings list"):
        google.embed_texts_google(["a", "b"], client=mock_client)


def test_google_batch_nested_embedding_shape() -> None:
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {
        "embeddings": [
            {"embedding": {"values": [0.3]}},
            {"embedding": {"values": [0.4]}},
        ],
    }
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    result = google.embed_texts_google(["a", "b"], client=mock_client)
    assert result == [[0.3], [0.4]]


def test_google_batch_count_mismatch_raises() -> None:
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"embeddings": [{"values": [0.1]}]}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with pytest.raises(RuntimeError, match="returned 1 vectors for 2 texts"):
        google.embed_texts_google(["a", "b"], client=mock_client)


def test_google_batch_entry_missing_values_raises() -> None:
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"embeddings": [{"foo": "bar"}]}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response

    with pytest.raises(RuntimeError, match="entry missing embedding values"):
        google.embed_texts_google(["a", "b"], client=mock_client)


def test_google_http_status_error_on_final_attempt() -> None:
    failing = MagicMock(status_code=500)
    failing.raise_for_status.side_effect = httpx.HTTPStatusError(
        "error", request=MagicMock(), response=failing
    )
    mock_client = MagicMock()
    mock_client.post.return_value = failing
    with patch("ingest.providers.google.MAX_RETRIES", 1), pytest.raises(httpx.HTTPStatusError):
        google.embed_texts_google(["x"], client=mock_client)


def test_google_request_error_on_final_attempt() -> None:
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.RequestError("network")
    with patch("ingest.providers.google.MAX_RETRIES", 1), pytest.raises(httpx.RequestError):
        google.embed_texts_google(["x"], client=mock_client)


def test_google_retry_loop_exhaustion() -> None:
    mock_client = MagicMock()
    mock_client.post.return_value = MagicMock(status_code=429)
    with patch("ingest.providers.google.MAX_RETRIES", 0), pytest.raises(RuntimeError):
        google._embed_batch_with_retry(
            mock_client,
            ["x", "y"],
            model="gemini-embedding-001",
            task_type="RETRIEVAL_DOCUMENT",
            dimensions=768,
        )


def test_google_batch_request_error_retries_before_success() -> None:
    success = MagicMock(status_code=200)
    success.json.return_value = {"embeddings": [{"values": [0.6]}, {"values": [0.7]}]}
    mock_client = MagicMock()
    mock_client.post.side_effect = [httpx.RequestError("network"), success]

    with patch("ingest.providers.google.time.sleep"):
        result = google.embed_texts_google(["a", "b"], client=mock_client)
    assert result == [[0.6], [0.7]]


def test_google_batch_raises_on_final_rate_limit() -> None:
    rate_limited = MagicMock(status_code=429)
    rate_limited.raise_for_status.side_effect = httpx.HTTPStatusError(
        "rate limited", request=MagicMock(), response=rate_limited
    )
    mock_client = MagicMock()
    mock_client.post.return_value = rate_limited
    with patch("ingest.providers.google.MAX_RETRIES", 1), pytest.raises(httpx.HTTPStatusError):
        google.embed_texts_google(["a", "b"], client=mock_client)


def test_google_batch_request_error_on_final_attempt() -> None:
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.RequestError("network")
    with patch("ingest.providers.google.MAX_RETRIES", 1), pytest.raises(httpx.RequestError):
        google.embed_texts_google(["a", "b"], client=mock_client)


def test_google_closes_owned_client() -> None:
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {"embedding": {"values": [0.1]}}
    mock_client = MagicMock()
    mock_client.post.return_value = mock_response
    with patch("ingest.providers.google.httpx.Client", return_value=mock_client):
        result = google.embed_texts_google(["x"])
    assert result == [[0.1]]
    mock_client.close.assert_called_once()


def test_google_single_http_status_error_retries_before_success() -> None:
    failing = MagicMock(status_code=404)
    failing.raise_for_status.side_effect = httpx.HTTPStatusError(
        "missing", request=MagicMock(), response=failing
    )
    success = MagicMock(status_code=200)
    success.json.return_value = {"embedding": {"values": [0.4]}}
    mock_client = MagicMock()
    mock_client.post.side_effect = [failing, success]

    with patch("ingest.providers.google.time.sleep"):
        result = google.embed_texts_google(["x"], client=mock_client)
    assert result == [[0.4]]


def test_google_batch_retries_429_before_success() -> None:
    rate_limited = MagicMock(status_code=429)
    success = MagicMock(status_code=200)
    success.json.return_value = {"embeddings": [{"values": [0.1]}, {"values": [0.2]}]}
    mock_client = MagicMock()
    mock_client.post.side_effect = [rate_limited, success]

    with patch("ingest.providers.google.time.sleep"):
        result = google.embed_texts_google(["a", "b"], client=mock_client)
    assert result == [[0.1], [0.2]]


def test_google_batch_http_status_error_retries_before_success() -> None:
    failing = MagicMock(status_code=404)
    failing.raise_for_status.side_effect = httpx.HTTPStatusError(
        "missing", request=MagicMock(), response=failing
    )
    success = MagicMock(status_code=200)
    success.json.return_value = {"embeddings": [{"values": [0.6]}, {"values": [0.7]}]}
    mock_client = MagicMock()
    mock_client.post.side_effect = [failing, success]

    with patch("ingest.providers.google.time.sleep"):
        result = google.embed_texts_google(["a", "b"], client=mock_client)
    assert result == [[0.6], [0.7]]


def test_google_single_retry_loop_exhaustion() -> None:
    mock_client = MagicMock()
    mock_client.post.return_value = MagicMock(status_code=429)
    with patch("ingest.providers.google.MAX_RETRIES", 0), pytest.raises(RuntimeError):
        google._embed_single_with_retry(
            mock_client,
            "x",
            model="gemini-embedding-001",
            task_type="RETRIEVAL_QUERY",
            dimensions=768,
        )

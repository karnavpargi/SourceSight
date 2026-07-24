from unittest.mock import patch

import pytest

from ingest.embed import embed_query, embed_texts


def test_embed_texts_routes_to_google(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ingest.embed.settings.embedding_provider", "google")
    vectors = [[0.1] * 768]
    with patch("ingest.embed.embed_texts_google", return_value=vectors) as mock_google:
        result = embed_texts(["hello"], task_type="RETRIEVAL_DOCUMENT")

    mock_google.assert_called_once_with(["hello"], task_type="RETRIEVAL_DOCUMENT")
    assert result == vectors


def test_embed_texts_routes_to_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ingest.embed.settings.embedding_provider", "ollama")
    vectors = [[0.1] * 768]
    with patch("ingest.embed.embed_texts_ollama", return_value=vectors) as mock_ollama:
        result = embed_texts(["hello"])

    mock_ollama.assert_called_once_with(["hello"])
    assert result == vectors


def test_embed_texts_returns_empty_when_provider_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ingest.embed.settings.embedding_provider", "none")
    with patch("ingest.embed.embed_texts_google") as mock_google, patch(
        "ingest.embed.embed_texts_ollama"
    ) as mock_ollama:
        assert embed_texts(["hello"]) == []

    mock_google.assert_not_called()
    mock_ollama.assert_not_called()


def test_embed_texts_validates_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ingest.embed.settings.embedding_provider", "ollama")
    with patch("ingest.embed.embed_texts_ollama", return_value=[[0.1] * 384]):
        with pytest.raises(ValueError, match="returned 384 dimensions"):
            embed_texts(["hello"])


def test_embed_texts_empty_input() -> None:
    with patch("ingest.embed.embed_texts_google", return_value=[]) as mock_google:
        assert embed_texts([]) == []

    mock_google.assert_not_called()


def test_embed_query_uses_retrieval_query_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ingest.embed.settings.embedding_provider", "google")
    with patch(
        "ingest.embed.embed_texts",
        return_value=[[0.2] * 768],
    ) as mock_embed_texts:
        vector = embed_query("what is revenue?")

    mock_embed_texts.assert_called_once_with(
        ["what is revenue?"],
        task_type="RETRIEVAL_QUERY",
    )
    assert vector == [0.2] * 768


def test_embed_query_raises_when_provider_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ingest.embed.settings.embedding_provider", "none")
    with pytest.raises(RuntimeError, match="EMBEDDING_PROVIDER=none"):
        embed_query("hello")

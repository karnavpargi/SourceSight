from __future__ import annotations

from unittest.mock import patch

import pytest

from ingest import embed


def test_embed_texts_empty_input() -> None:
    assert embed.embed_texts([]) == []


def test_embed_texts_routes_to_ollama() -> None:
    vectors = [[0.1] * 768]
    with patch("ingest.embed.settings.embedding_provider", "ollama"), patch(
        "ingest.embed.settings.embedding_dimensions", 768
    ), patch("ingest.embed.embed_texts_ollama", return_value=vectors) as mock_ollama:
        result = embed.embed_texts(["hello"])
    mock_ollama.assert_called_once_with(["hello"])
    assert result == vectors


def test_embed_texts_routes_to_openai() -> None:
    vectors = [[0.2] * 768]
    with patch("ingest.embed.settings.embedding_provider", "openai"), patch(
        "ingest.embed.settings.embedding_dimensions", 768
    ), patch("ingest.embed.embed_texts_openai", return_value=vectors) as mock_openai:
        result = embed.embed_texts(["hello"])
    mock_openai.assert_called_once_with(["hello"])
    assert result == vectors


def test_embed_texts_unsupported_provider() -> None:
    with patch("ingest.embed.settings.embedding_provider", "unknown"):
        with pytest.raises(ValueError, match="Unsupported EMBEDDING_PROVIDER"):
            embed.embed_texts(["hello"])


def test_embed_texts_dimension_mismatch() -> None:
    with patch("ingest.embed.settings.embedding_provider", "ollama"), patch(
        "ingest.embed.settings.embedding_dimensions", 768
    ), patch("ingest.embed.embed_texts_ollama", return_value=[[0.1] * 384]):
        with pytest.raises(ValueError, match="returned 384-dim vectors"):
            embed.embed_texts(["hello"])

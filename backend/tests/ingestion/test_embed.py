from unittest.mock import patch

import pytest

from ingest.embed import embed_texts


def test_embed_texts_routes_to_ollama() -> None:
    vectors = [[0.1] * 768]
    with patch("ingest.embed.embed_texts_ollama", return_value=vectors) as mock_ollama:
        result = embed_texts(["hello"])

    mock_ollama.assert_called_once_with(["hello"])
    assert result == vectors


def test_embed_texts_validates_dimensions() -> None:
    with patch("ingest.embed.embed_texts_ollama", return_value=[[0.1] * 384]):
        with pytest.raises(ValueError, match="Ollama returned 384-dim"):
            embed_texts(["hello"])


def test_embed_texts_empty_input() -> None:
    with patch("ingest.embed.embed_texts_ollama", return_value=[]) as mock_ollama:
        assert embed_texts([]) == []

    mock_ollama.assert_called_once_with([])

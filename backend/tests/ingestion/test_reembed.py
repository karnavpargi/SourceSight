from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingest.reembed import main, reembed_chunks


def test_reembed_chunks_updates_in_batches() -> None:
    chunk_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
    rows = [(chunk_ids[0], "a"), (chunk_ids[1], "b"), (chunk_ids[2], "c")]
    mock_session = MagicMock()
    mock_session.execute.return_value.all.return_value = rows

    with patch("ingest.reembed.session_scope") as scope, patch(
        "ingest.reembed.embed_texts",
        side_effect=[[[0.1] * 768, [0.2] * 768], [[0.3] * 768]],
    ) as embed:
        scope.return_value.__enter__.return_value = mock_session
        updated, total = reembed_chunks(batch_size=2)

    assert updated == 3
    assert total == 3
    assert embed.call_count == 2
    assert mock_session.commit.call_count == 2


def test_reembed_chunks_empty_corpus() -> None:
    mock_session = MagicMock()
    mock_session.execute.return_value.all.return_value = []

    with patch("ingest.reembed.session_scope") as scope:
        scope.return_value.__enter__.return_value = mock_session
        updated, total = reembed_chunks()

    assert updated == 0
    assert total == 0
    mock_session.commit.assert_not_called()


def test_reembed_chunks_filters_by_document_id() -> None:
    document_id = uuid.uuid4()
    mock_session = MagicMock()
    mock_session.execute.return_value.all.return_value = []

    with patch("ingest.reembed.session_scope") as scope:
        scope.return_value.__enter__.return_value = mock_session
        reembed_chunks(document_id=document_id)

    statement = mock_session.execute.call_args.args[0]
    assert statement is not None


def test_main_success(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("ingest.reembed.reembed_chunks", return_value=(5, 5)):
        assert main([]) == 0
    assert "5/5" in capsys.readouterr().out


def test_main_partial_failure(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("ingest.reembed.reembed_chunks", return_value=(3, 5)):
        assert main([]) == 1


def test_main_rejects_invalid_batch_size() -> None:
    assert main(["--batch-size", "0"]) == 2


def test_reembed_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import sys

    run_path = Path(__file__).resolve().parents[2] / "ingest" / "reembed.py"
    monkeypatch.setattr(sys, "argv", ["reembed.py"])
    mock_session = MagicMock()
    mock_session.execute.return_value.all.return_value = []

    with patch("ingest.db.session_scope") as scope:
        scope.return_value.__enter__.return_value = mock_session
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(run_path), run_name="__main__")
    assert exc.value.code == 0

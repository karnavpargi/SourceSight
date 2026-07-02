from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingest import db


def test_get_database_url_rewrites_postgresql_scheme() -> None:
    with patch("ingest.db.settings.database_url", "postgresql://user:pass@host/db"):
        assert db.get_database_url() == "postgresql+psycopg://user:pass@host/db"


def test_get_database_url_passthrough() -> None:
    with patch("ingest.db.settings.database_url", "postgresql+psycopg://user:pass@host/db"):
        assert db.get_database_url() == "postgresql+psycopg://user:pass@host/db"


def test_session_scope_commits_on_success() -> None:
    session = MagicMock()
    factory = MagicMock(return_value=session)
    with patch("ingest.db.get_session_factory", return_value=factory):
        with db.session_scope() as scoped:
            assert scoped is session
    session.commit.assert_called_once()
    session.close.assert_called_once()


def test_session_scope_rolls_back_on_error() -> None:
    session = MagicMock()
    factory = MagicMock(return_value=session)
    with patch("ingest.db.get_session_factory", return_value=factory):
        with pytest.raises(RuntimeError), db.session_scope():
            raise RuntimeError("boom")
    session.rollback.assert_called_once()
    session.close.assert_called_once()


def test_get_engine_and_factory_are_cached() -> None:
    db._engine = None
    db._session_factory = None
    mock_engine = MagicMock()
    with patch("ingest.db.create_engine", return_value=mock_engine) as create_engine:
        assert db.get_engine() is mock_engine
        assert db.get_engine() is mock_engine
        create_engine.assert_called_once()
        factory = db.get_session_factory()
        assert factory.kw["bind"] is mock_engine
        assert db.get_session_factory() is factory
    db._engine = None
    db._session_factory = None

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.database import session as db_session


def test_get_database_url_rewrites_postgresql_scheme() -> None:
    with patch("app.database.session.settings.database_url", "postgresql://user:pass@host/db"):
        assert db_session.get_database_url() == "postgresql+psycopg://user:pass@host/db"


def test_get_database_url_passthrough() -> None:
    with patch(
        "app.database.session.settings.database_url",
        "postgresql+psycopg://user:pass@host/db",
    ):
        assert db_session.get_database_url() == "postgresql+psycopg://user:pass@host/db"


def test_session_scope_commits_on_success() -> None:
    session = MagicMock()
    factory = MagicMock(return_value=session)
    with patch("app.database.session.get_session_factory", return_value=factory):
        with db_session.session_scope() as scoped:
            assert scoped is session
    session.commit.assert_called_once()
    session.close.assert_called_once()


def test_session_scope_rolls_back_on_error() -> None:
    session = MagicMock()
    factory = MagicMock(return_value=session)
    with patch("app.database.session.get_session_factory", return_value=factory):
        with pytest.raises(RuntimeError), db_session.session_scope():
            raise RuntimeError("boom")
    session.rollback.assert_called_once()
    session.close.assert_called_once()


def test_get_engine_and_factory_are_cached() -> None:
    db_session._engine = None
    db_session._session_factory = None
    mock_engine = MagicMock()
    with patch("app.database.session.create_engine", return_value=mock_engine) as create_engine:
        assert db_session.get_engine() is mock_engine
        assert db_session.get_engine() is mock_engine
        create_engine.assert_called_once()
        factory = db_session.get_session_factory()
        assert factory.kw["bind"] is mock_engine
        assert db_session.get_session_factory() is factory
    db_session._engine = None
    db_session._session_factory = None

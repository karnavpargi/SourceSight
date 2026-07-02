from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from postgrest.exceptions import APIError

from app.database.profiles import ensure_profile

USER_ID = UUID("550e8400-e29b-41d4-a716-446655440000")
EMAIL = "analyst@example.com"


def _profiles_client(*, execute_results: list) -> MagicMock:
    table = MagicMock()
    table.select.return_value = table
    table.eq.return_value = table
    table.maybe_single.return_value = table
    table.insert.return_value = table
    table.execute = AsyncMock(side_effect=execute_results)

    client = MagicMock()
    client.table.return_value = table
    return client, table


def test_ensure_profile_skips_when_row_exists() -> None:
    existing = MagicMock()
    existing.data = {"id": str(USER_ID)}
    client, table = _profiles_client(execute_results=[existing])

    asyncio.run(ensure_profile(client, user_id=USER_ID, email=EMAIL))

    table.insert.assert_not_called()


def test_ensure_profile_inserts_when_missing() -> None:
    client, table = _profiles_client(execute_results=[None, MagicMock()])

    asyncio.run(ensure_profile(client, user_id=USER_ID, email=EMAIL))

    table.insert.assert_called_once_with({"id": str(USER_ID), "email": EMAIL})


def test_ensure_profile_ignores_duplicate_insert_race() -> None:
    duplicate = APIError(
        {
            "message": "duplicate key value violates unique constraint",
            "code": "23505",
            "hint": None,
            "details": None,
        }
    )
    client, _ = _profiles_client(execute_results=[None, duplicate])

    asyncio.run(ensure_profile(client, user_id=USER_ID, email=EMAIL))


def test_ensure_profile_reraises_non_duplicate_insert_errors() -> None:
    denied = APIError(
        {
            "message": "permission denied",
            "code": "42501",
            "hint": None,
            "details": None,
        }
    )
    client, _ = _profiles_client(execute_results=[None, denied])

    with pytest.raises(APIError):
        asyncio.run(ensure_profile(client, user_id=USER_ID, email=EMAIL))

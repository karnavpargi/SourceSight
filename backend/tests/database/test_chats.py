from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from app.database.chats import (
    ChatForbiddenError,
    ChatNotFoundError,
    create_thread,
    get_thread,
    list_threads,
)

USER_ID = UUID("550e8400-e29b-41d4-a716-446655440000")
OTHER_USER_ID = UUID("660e8400-e29b-41d4-a716-446655440001")
THREAD_ID = UUID("770e8400-e29b-41d4-a716-446655440002")
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


def _thread_row(*, user_id: UUID = USER_ID) -> dict:
    return {
        "id": str(THREAD_ID),
        "user_id": str(user_id),
        "title": "AWS segment review",
        "created_at": NOW.isoformat(),
        "updated_at": NOW.isoformat(),
    }


def _mock_table_chain(*, data, maybe_single_data=None):
    table = MagicMock()
    table.select.return_value = table
    table.eq.return_value = table
    table.order.return_value = table
    table.insert.return_value = table
    table.single.return_value = table
    table.maybe_single.return_value = table

    execute_result = MagicMock()
    execute_result.data = data
    table.execute = AsyncMock(return_value=execute_result)

    if maybe_single_data is not None:
        maybe_result = MagicMock()
        maybe_result.data = maybe_single_data
        table.execute = AsyncMock(return_value=maybe_result)

    client = MagicMock()
    client.table.return_value = table
    return client, table


def test_list_threads_returns_parsed_records() -> None:
    client, table = _mock_table_chain(data=[_thread_row()])

    threads = asyncio.run(list_threads(client, USER_ID))

    assert len(threads) == 1
    assert threads[0].id == THREAD_ID
    assert threads[0].title == "AWS segment review"
    table.eq.assert_called_with("user_id", str(USER_ID))


def test_create_thread_inserts_and_returns_record() -> None:
    client, table = _mock_table_chain(data=_thread_row())
    table.insert.return_value = table
    table.select.return_value = table
    table.single.return_value = table

    thread = asyncio.run(create_thread(client, USER_ID, "New thread"))

    assert thread.user_id == USER_ID
    table.insert.assert_called_once_with({"user_id": str(USER_ID), "title": "New thread"})


def test_get_thread_returns_owned_thread() -> None:
    client, _ = _mock_table_chain(data=None, maybe_single_data=_thread_row())

    thread = asyncio.run(get_thread(client, USER_ID, THREAD_ID))

    assert thread.id == THREAD_ID


def test_get_thread_raises_forbidden_when_owned_by_other_user() -> None:
    client, _ = _mock_table_chain(data=None, maybe_single_data=_thread_row(user_id=OTHER_USER_ID))

    with pytest.raises(ChatForbiddenError):
        asyncio.run(get_thread(client, USER_ID, THREAD_ID))


def test_get_thread_raises_not_found_when_missing() -> None:
    user_client, _ = _mock_table_chain(data=None, maybe_single_data=None)
    admin_client, _ = _mock_table_chain(data=None, maybe_single_data=None)

    with patch(
        "app.database.chats.create_admin_client",
        new=AsyncMock(return_value=admin_client),
    ):
        with pytest.raises(ChatNotFoundError):
            asyncio.run(get_thread(user_client, USER_ID, THREAD_ID))

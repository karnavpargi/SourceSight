from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from supabase_auth.errors import AuthApiError

from app.database.chats import ChatForbiddenError, ChatMessageRecord, ChatNotFoundError, ChatThreadRecord
from app.main import create_app

USER_A_ID = UUID("550e8400-e29b-41d4-a716-446655440000")
USER_B_ID = UUID("660e8400-e29b-41d4-a716-446655440001")
THREAD_ID = UUID("770e8400-e29b-41d4-a716-446655440002")
MESSAGE_ID = UUID("880e8400-e29b-41d4-a716-446655440003")
VALID_TOKEN = "valid-token"
EXPIRED_TOKEN = "expired-token"
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture
def thread_record() -> ChatThreadRecord:
    return ChatThreadRecord(
        id=THREAD_ID,
        user_id=USER_A_ID,
        title="AWS segment review",
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.fixture
def message_record() -> ChatMessageRecord:
    return ChatMessageRecord(
        id=MESSAGE_ID,
        thread_id=THREAD_ID,
        role="user",
        content="AWS operating income",
        message_data=None,
        created_at=NOW,
    )


def _mock_user_response(*, user_id: str = str(USER_A_ID), email: str = "analyst@example.com") -> MagicMock:
    response = MagicMock()
    response.user.id = user_id
    response.user.email = email
    return response


def _auth_headers(token: str = VALID_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_list_threads_requires_authentication(client: TestClient) -> None:
    response = client.get("/threads")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


def test_list_threads_rejects_expired_token(client: TestClient) -> None:
    mock_supabase = MagicMock()
    mock_supabase.auth.get_user = AsyncMock(side_effect=AuthApiError("Invalid JWT", 401, "bad_jwt"))

    with patch("app.auth.dependencies.create_user_client", new=AsyncMock(return_value=mock_supabase)):
        response = client.get("/threads", headers=_auth_headers(EXPIRED_TOKEN))

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired token"


def test_list_threads_returns_threads_for_valid_token(
    client: TestClient,
    thread_record: ChatThreadRecord,
) -> None:
    mock_supabase = MagicMock()
    mock_supabase.auth.get_user = AsyncMock(return_value=_mock_user_response())

    with patch("app.auth.dependencies.create_user_client", new=AsyncMock(return_value=mock_supabase)), patch(
        "app.api.chat.chat_store.list_threads",
        new=AsyncMock(return_value=[thread_record]),
    ):
        response = client.get("/threads", headers=_auth_headers())

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["id"] == str(THREAD_ID)
    assert payload[0]["title"] == "AWS segment review"


def test_create_thread_returns_created_thread(
    client: TestClient,
    thread_record: ChatThreadRecord,
) -> None:
    mock_supabase = MagicMock()
    mock_supabase.auth.get_user = AsyncMock(return_value=_mock_user_response())

    with patch("app.auth.dependencies.create_user_client", new=AsyncMock(return_value=mock_supabase)), patch(
        "app.api.chat.chat_store.create_thread",
        new=AsyncMock(return_value=thread_record),
    ) as create_thread:
        response = client.post(
            "/threads",
            headers=_auth_headers(),
            json={"title": "AWS segment review"},
        )

    assert response.status_code == 201
    create_thread.assert_awaited_once()
    assert response.json()["title"] == "AWS segment review"


def test_list_thread_messages_returns_403_for_other_users_thread(client: TestClient) -> None:
    mock_supabase = MagicMock()
    mock_supabase.auth.get_user = AsyncMock(return_value=_mock_user_response())

    with patch("app.auth.dependencies.create_user_client", new=AsyncMock(return_value=mock_supabase)), patch(
        "app.api.chat.chat_store.list_thread_messages",
        new=AsyncMock(side_effect=ChatForbiddenError(str(THREAD_ID))),
    ):
        response = client.get(
            f"/threads/{THREAD_ID}/messages",
            headers=_auth_headers(),
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden"


def test_list_thread_messages_returns_404_for_missing_thread(client: TestClient) -> None:
    mock_supabase = MagicMock()
    mock_supabase.auth.get_user = AsyncMock(return_value=_mock_user_response())

    with patch("app.auth.dependencies.create_user_client", new=AsyncMock(return_value=mock_supabase)), patch(
        "app.api.chat.chat_store.list_thread_messages",
        new=AsyncMock(side_effect=ChatNotFoundError(str(THREAD_ID))),
    ):
        response = client.get(
            f"/threads/{THREAD_ID}/messages",
            headers=_auth_headers(),
        )

    assert response.status_code == 404


def test_stream_chat_requires_authentication(client: TestClient) -> None:
    response = client.post(
        "/chat/stream",
        json={"threadId": str(THREAD_ID), "messages": []},
    )
    assert response.status_code == 401


def test_stream_chat_delegates_to_orchestrator(
    client: TestClient,
    thread_record: ChatThreadRecord,
) -> None:
    mock_supabase = MagicMock()
    mock_supabase.auth.get_user = AsyncMock(return_value=_mock_user_response())
    mock_response = MagicMock(status_code=200)
    mock_response.headers = {"content-type": "text/event-stream"}

    with patch("app.auth.dependencies.create_user_client", new=AsyncMock(return_value=mock_supabase)), patch(
        "app.api.chat.chat_store.get_thread",
        new=AsyncMock(return_value=thread_record),
    ), patch(
        "app.api.chat.run_chat_turn",
        new=AsyncMock(return_value=mock_response),
    ) as run_chat_turn:
        response = client.post(
            "/chat/stream",
            headers=_auth_headers(),
            json={
                "threadId": str(THREAD_ID),
                "messages": [
                    {
                        "id": "user-1",
                        "role": "user",
                        "parts": [{"type": "text", "text": "AWS operating income"}],
                    }
                ],
            },
        )

    assert response.status_code == 200
    run_chat_turn.assert_awaited_once()
    assert run_chat_turn.await_args.kwargs["user_text"] == "AWS operating income"


def test_user_a_cannot_access_user_b_thread_messages(client: TestClient) -> None:
    mock_supabase = MagicMock()
    mock_supabase.auth.get_user = AsyncMock(
        return_value=_mock_user_response(user_id=str(USER_A_ID), email="a@example.com"),
    )

    with patch("app.auth.dependencies.create_user_client", new=AsyncMock(return_value=mock_supabase)), patch(
        "app.api.chat.chat_store.list_thread_messages",
        new=AsyncMock(side_effect=ChatForbiddenError(str(THREAD_ID))),
    ):
        response = client.get(f"/threads/{THREAD_ID}/messages", headers=_auth_headers())

    assert response.status_code == 403

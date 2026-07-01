from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from supabase_auth.errors import AuthApiError

from app.auth.dependencies import get_current_user
from app.auth.types import CurrentUser

USER_ID = "550e8400-e29b-41d4-a716-446655440000"
USER_EMAIL = "analyst@example.com"


@pytest.fixture
def auth_app() -> FastAPI:
    app = FastAPI()

    @app.get("/me")
    async def me(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        return user

    return app


def _mock_user_response(*, email: str | None = USER_EMAIL) -> MagicMock:
    response = MagicMock()
    response.user.id = USER_ID
    response.user.email = email
    return response


def test_get_current_user_missing_authorization(auth_app: FastAPI) -> None:
    response = TestClient(auth_app).get("/me")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


def test_get_current_user_invalid_scheme(auth_app: FastAPI) -> None:
    response = TestClient(auth_app).get(
        "/me",
        headers={"Authorization": "Basic abc123"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"


def test_get_current_user_invalid_token(auth_app: FastAPI) -> None:
    mock_client = MagicMock()
    mock_client.auth.get_user = AsyncMock(
        side_effect=AuthApiError("Invalid JWT", 401, "bad_jwt"),
    )

    with patch(
        "app.auth.dependencies.create_user_client",
        new=AsyncMock(return_value=mock_client),
    ):
        response = TestClient(auth_app).get(
            "/me",
            headers={"Authorization": "Bearer bad-token"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired token"
    mock_client.auth.get_user.assert_awaited_once_with("bad-token")


def test_get_current_user_missing_user(auth_app: FastAPI) -> None:
    mock_client = MagicMock()
    mock_client.auth.get_user = AsyncMock(return_value=None)

    with patch(
        "app.auth.dependencies.create_user_client",
        new=AsyncMock(return_value=mock_client),
    ):
        response = TestClient(auth_app).get(
            "/me",
            headers={"Authorization": "Bearer valid-token"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired token"


def test_get_current_user_missing_email(auth_app: FastAPI) -> None:
    mock_client = MagicMock()
    mock_client.auth.get_user = AsyncMock(return_value=_mock_user_response(email=None))

    with patch(
        "app.auth.dependencies.create_user_client",
        new=AsyncMock(return_value=mock_client),
    ):
        response = TestClient(auth_app).get(
            "/me",
            headers={"Authorization": "Bearer valid-token"},
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired token"


def test_get_current_user_valid_token(auth_app: FastAPI) -> None:
    mock_client = MagicMock()
    mock_client.auth.get_user = AsyncMock(return_value=_mock_user_response())

    with patch(
        "app.auth.dependencies.create_user_client",
        new=AsyncMock(return_value=mock_client),
    ) as create_user_client:
        response = TestClient(auth_app).get(
            "/me",
            headers={"Authorization": "Bearer valid-token"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "id": USER_ID,
        "email": USER_EMAIL,
    }
    create_user_client.assert_awaited_once_with("valid-token")
    mock_client.auth.get_user.assert_awaited_once_with("valid-token")

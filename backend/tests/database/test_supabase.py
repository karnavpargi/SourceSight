from unittest.mock import AsyncMock, patch

import pytest

from app.database import supabase as supabase_module


@pytest.fixture(autouse=True)
def reset_admin_client() -> None:
    supabase_module._admin_client = None
    yield
    supabase_module._admin_client = None


@pytest.mark.anyio
async def test_create_user_client_uses_anon_key_and_bearer() -> None:
    mock_client = object()
    with patch(
        "app.database.supabase.acreate_client",
        new=AsyncMock(return_value=mock_client),
    ) as acreate_client:
        result = await supabase_module.create_user_client("jwt-token")

    assert result is mock_client
    acreate_client.assert_awaited_once()
    url, key = acreate_client.await_args.args
    options = acreate_client.await_args.kwargs["options"]
    assert url == supabase_module.settings.supabase_url
    assert key == supabase_module.settings.supabase_anon_key
    assert options.headers["Authorization"] == "Bearer jwt-token"
    assert options.auto_refresh_token is False
    assert options.persist_session is False


@pytest.mark.anyio
async def test_create_user_client_preserves_existing_bearer_prefix() -> None:
    with patch(
        "app.database.supabase.acreate_client",
        new=AsyncMock(return_value=object()),
    ) as acreate_client:
        await supabase_module.create_user_client("Bearer jwt-token")

    options = acreate_client.await_args.kwargs["options"]
    assert options.headers["Authorization"] == "Bearer jwt-token"


@pytest.mark.anyio
async def test_create_admin_client_uses_service_role_key() -> None:
    mock_client = object()
    with patch(
        "app.database.supabase.acreate_client",
        new=AsyncMock(return_value=mock_client),
    ) as acreate_client:
        first = await supabase_module.create_admin_client()
        second = await supabase_module.create_admin_client()

    assert first is mock_client
    assert second is mock_client
    acreate_client.assert_awaited_once()
    url, key = acreate_client.await_args.args
    options = acreate_client.await_args.kwargs["options"]
    assert url == supabase_module.settings.supabase_url
    assert key == supabase_module.settings.supabase_service_role_key
    assert options.auto_refresh_token is False
    assert options.persist_session is False

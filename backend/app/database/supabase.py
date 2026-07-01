import asyncio

from supabase import AsyncClient, acreate_client
from supabase.lib.client_options import DEFAULT_HEADERS, AsyncClientOptions

from app.config import settings

_admin_client: AsyncClient | None = None
_admin_client_lock = asyncio.Lock()


def _bearer_header(access_token: str) -> str:
    prefix = "Bearer "
    if access_token.startswith(prefix):
        return access_token
    return f"{prefix}{access_token}"


def _server_options(*, authorization: str | None = None) -> AsyncClientOptions:
    headers = DEFAULT_HEADERS.copy()
    if authorization is not None:
        headers["Authorization"] = authorization
    return AsyncClientOptions(
        headers=headers,
        auto_refresh_token=False,
        persist_session=False,
    )


async def create_user_client(access_token: str) -> AsyncClient:
    """Return a Supabase client scoped to the user's JWT (anon key + bearer)."""
    return await acreate_client(
        settings.supabase_url,
        settings.supabase_anon_key,
        options=_server_options(authorization=_bearer_header(access_token)),
    )


async def create_admin_client() -> AsyncClient:
    """Return a Supabase client with the service role key (bypasses RLS)."""
    global _admin_client

    if _admin_client is not None:
        return _admin_client

    async with _admin_client_lock:
        if _admin_client is None:
            _admin_client = await acreate_client(
                settings.supabase_url,
                settings.supabase_service_role_key,
                options=_server_options(),
            )
        return _admin_client

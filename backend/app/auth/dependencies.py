from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase import AsyncClient
from supabase_auth.errors import AuthApiError

from app.auth.types import CurrentUser
from app.database.supabase import create_user_client

_bearer = HTTPBearer(auto_error=False)


def _require_bearer_token(credentials: HTTPAuthorizationCredentials | None) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return credentials.credentials


async def get_access_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    return _require_bearer_token(credentials)


async def get_user_client(access_token: str = Depends(get_access_token)) -> AsyncClient:
    return await create_user_client(access_token)


async def get_current_user(access_token: str = Depends(get_access_token)) -> CurrentUser:
    client = await create_user_client(access_token)

    try:
        user_response = await client.auth.get_user(access_token)
    except AuthApiError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from None

    if user_response is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user = user_response.user
    if user.email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    return CurrentUser(id=UUID(user.id), email=user.email)

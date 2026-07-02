"""Profile row helpers for Supabase-authenticated users."""

from __future__ import annotations

from uuid import UUID

from postgrest.exceptions import APIError
from supabase import AsyncClient


async def ensure_profile(client: AsyncClient, *, user_id: UUID, email: str) -> None:
    """Create the user's profile row on first API use if it does not exist yet."""
    response = (
        await client.table("profiles")
        .select("id")
        .eq("id", str(user_id))
        .maybe_single()
        .execute()
    )
    if response is not None:
        return

    try:
        await client.table("profiles").insert({"id": str(user_id), "email": email}).execute()
    except APIError as exc:
        # Another concurrent request may have created the row first.
        if exc.code != "23505":
            raise

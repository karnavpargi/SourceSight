"""Shared httpx clients with HTTP/2 enabled (falls back to HTTP/1.1 when unsupported)."""

from __future__ import annotations

import httpx

_sync_client: httpx.Client | None = None
_async_client: httpx.AsyncClient | None = None


def get_sync_client() -> httpx.Client:
    global _sync_client
    if _sync_client is None:
        _sync_client = httpx.Client(http2=True, timeout=120.0)
    return _sync_client


def get_async_client() -> httpx.AsyncClient:
    global _async_client
    if _async_client is None:
        _async_client = httpx.AsyncClient(http2=True, timeout=120.0)
    return _async_client


def http_get(url: str, **kwargs: object) -> httpx.Response:
    return get_sync_client().get(url, **kwargs)


async def close_http_clients() -> None:
    global _sync_client, _async_client
    if _async_client is not None:
        await _async_client.aclose()
        _async_client = None
    if _sync_client is not None:
        _sync_client.close()
        _sync_client = None

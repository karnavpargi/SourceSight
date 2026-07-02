import pytest

from app.config import Settings


def test_chat_provider_accepts_local_alias_ollama() -> None:
    settings = Settings(
        supabase_url="https://example.supabase.co",
        supabase_anon_key="anon",
        supabase_service_role_key="service",
        database_url="postgresql://postgres:pw@localhost:5432/postgres",
        allowed_origins=["http://localhost:5173"],
        chat_provider="ollama",
        google_api_key="unused",
    )
    assert settings.chat_provider == "local"


def test_chat_provider_requires_google_key() -> None:
    with pytest.raises(ValueError, match="GOOGLE_API_KEY is required"):
        Settings(
            supabase_url="https://example.supabase.co",
            supabase_anon_key="anon",
            supabase_service_role_key="service",
            database_url="postgresql://postgres:pw@localhost:5432/postgres",
            allowed_origins=["http://localhost:5173"],
            chat_provider="google",
            google_api_key="",
        )


def test_chat_provider_requires_opencode_key() -> None:
    with pytest.raises(ValueError, match="OPENCODE_API_KEY is required"):
        Settings(
            supabase_url="https://example.supabase.co",
            supabase_anon_key="anon",
            supabase_service_role_key="service",
            database_url="postgresql://postgres:pw@localhost:5432/postgres",
            allowed_origins=["http://localhost:5173"],
            chat_provider="opencode",
            opencode_api_key="",
        )

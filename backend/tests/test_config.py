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
        use_ollama=True,
        google_api_key="unused",
        _env_file=None,
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
            _env_file=None,
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
            _env_file=None,
        )


def test_chat_provider_rejects_local_when_ollama_disabled() -> None:
    with pytest.raises(ValueError, match="USE_OLLAMA is false"):
        Settings(
            supabase_url="https://example.supabase.co",
            supabase_anon_key="anon",
            supabase_service_role_key="service",
            database_url="postgresql://postgres:pw@localhost:5432/postgres",
            allowed_origins=["http://localhost:5173"],
            chat_provider="local",
            use_ollama=False,
            google_api_key="unused",
            _env_file=None,
        )

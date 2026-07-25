from pathlib import Path

import pytest

from app.config import Settings


def test_dotenv_overrides_shell_google_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "SUPABASE_URL=https://example.supabase.co",
                "SUPABASE_ANON_KEY=anon",
                "SUPABASE_SERVICE_ROLE_KEY=service",
                "DATABASE_URL=postgresql://postgres:pw@localhost:5432/postgres",
                "ALLOWED_ORIGINS=http://localhost:5173",
                "CHAT_PROVIDER=google",
                "EMBEDDING_PROVIDER=google",
                "GOOGLE_API_KEY=from-dotenv-key",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "from-shell-stale-key")

    settings = Settings(_env_file=env_file)

    assert settings.google_api_key == "from-dotenv-key"


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


def test_embedding_provider_requires_google_key() -> None:
    with pytest.raises(ValueError, match="GOOGLE_API_KEY is required when EMBEDDING_PROVIDER"):
        Settings(
            supabase_url="https://example.supabase.co",
            supabase_anon_key="anon",
            supabase_service_role_key="service",
            database_url="postgresql://postgres:pw@localhost:5432/postgres",
            allowed_origins=["http://localhost:5173"],
            chat_provider="opencode",
            opencode_api_key="zen-key",
            embedding_provider="google",
            google_api_key="",
            _env_file=None,
        )


def test_embedding_provider_accepts_none_without_google_key() -> None:
    settings = Settings(
        supabase_url="https://example.supabase.co",
        supabase_anon_key="anon",
        supabase_service_role_key="service",
        database_url="postgresql://postgres:pw@localhost:5432/postgres",
        allowed_origins=["http://localhost:5173"],
        chat_provider="opencode",
        opencode_api_key="zen-key",
        embedding_provider="none",
        google_api_key="",
        _env_file=None,
    )
    assert settings.embedding_provider == "none"


def test_router_model_defaults_to_flash_lite() -> None:
    settings = Settings(
        supabase_url="https://example.supabase.co",
        supabase_anon_key="anon",
        supabase_service_role_key="service",
        database_url="postgresql://postgres:pw@localhost:5432/postgres",
        allowed_origins=["http://localhost:5173"],
        chat_provider="google",
        google_api_key="key",
        _env_file=None,
    )
    assert settings.chat_router_model == "gemini-2.0-flash-lite"


def test_model_prices_parse_exact_model_ids() -> None:
    settings = Settings(
        supabase_url="https://example.supabase.co",
        supabase_anon_key="anon",
        supabase_service_role_key="service",
        database_url="postgresql://postgres:pw@localhost:5432/postgres",
        allowed_origins=["http://localhost:5173"],
        chat_provider="google",
        google_api_key="key",
        chat_model_prices={"gemini-3.5-flash-lite": (0.30, 2.50)},
        _env_file=None,
    )
    assert settings.chat_model_prices["gemini-3.5-flash-lite"] == (0.30, 2.50)


def test_embedding_provider_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="embedding_provider"):
        Settings(
            supabase_url="https://example.supabase.co",
            supabase_anon_key="anon",
            supabase_service_role_key="service",
            database_url="postgresql://postgres:pw@localhost:5432/postgres",
            allowed_origins=["http://localhost:5173"],
            chat_provider="opencode",
            opencode_api_key="zen-key",
            embedding_provider="openai",
            google_api_key="key",
            _env_file=None,
        )

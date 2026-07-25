from functools import lru_cache
from typing import Annotated, Literal, Self

from pydantic import field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

ChatProvider = Literal["local", "google", "opencode"]
CHAT_PROVIDERS: frozenset[str] = frozenset({"local", "google", "opencode"})

EmbeddingProvider = Literal["google", "ollama", "none"]
EMBEDDING_PROVIDERS: frozenset[str] = frozenset({"google", "ollama", "none"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Prefer backend/.env over exported shell vars. Stale GOOGLE_API_KEY in
        # ~/.zshrc otherwise overrides the operator .env and breaks chat/providers.
        del settings_cls
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    # Supabase (auth + API)
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str

    # Postgres (Alembic + direct DB access)
    database_url: str

    # Chat LLM — local (Ollama), Google AI Studio, or OpenCode Zen
    chat_provider: ChatProvider = "google"
    chat_model: str = "gemini-3.5-flash-lite"
    chat_router_model: str = "gemini-flash-lite-latest"
    # Values are paid-tier USD per million (input, output) tokens.
    chat_model_prices: dict[str, tuple[float, float]] = {}
    google_api_key: str = ""
    opencode_api_key: str = ""
    opencode_base_url: str = "https://opencode.ai/zen/go/v1"

    # Embeddings — Ollama (local or remote), Google Gemini, or FTS-only (none)
    embedding_provider: EmbeddingProvider = "google"
    embedding_dimensions: int = 768
    google_embedding_model: str = "gemini-embedding-001"
    use_ollama: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"

    # Server
    allowed_origins: Annotated[list[str], NoDecode]
    log_json: bool = True

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("chat_provider", mode="before")
    @classmethod
    def normalize_chat_provider(cls, value: str) -> str:
        provider = value.lower()
        if provider == "ollama":
            return "local"
        return provider

    @model_validator(mode="after")
    def require_provider_keys(self) -> Self:
        if self.chat_provider not in CHAT_PROVIDERS:
            supported = ", ".join(sorted(CHAT_PROVIDERS))
            raise ValueError(f"CHAT_PROVIDER must be one of: {supported}")

        if self.chat_provider == "google" and not self.google_api_key.strip():
            raise ValueError("GOOGLE_API_KEY is required when CHAT_PROVIDER is google")
        if self.chat_provider == "opencode" and not self.opencode_api_key.strip():
            raise ValueError("OPENCODE_API_KEY is required when CHAT_PROVIDER is opencode")
        if self.chat_provider == "local" and not self.use_ollama:
            raise ValueError("CHAT_PROVIDER cannot be local when USE_OLLAMA is false")
        if not self.chat_model.strip():
            raise ValueError("CHAT_MODEL is required")
        if not self.chat_router_model.strip():
            raise ValueError("CHAT_ROUTER_MODEL is required")
        if self.embedding_provider not in EMBEDDING_PROVIDERS:
            supported = ", ".join(sorted(EMBEDDING_PROVIDERS))
            raise ValueError(f"EMBEDDING_PROVIDER must be one of: {supported}")
        if self.embedding_provider == "google" and not self.google_api_key.strip():
            raise ValueError("GOOGLE_API_KEY is required when EMBEDDING_PROVIDER is google")
        return self

    def ollama_openai_base_url(self) -> str:
        base = self.ollama_base_url.rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

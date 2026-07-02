from functools import lru_cache
from typing import Annotated, Literal, Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

ChatProvider = Literal["local", "google", "opencode"]
CHAT_PROVIDERS: frozenset[str] = frozenset({"local", "google", "opencode"})


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Supabase (auth + API)
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str

    # Postgres (Alembic + direct DB access)
    database_url: str

    # Chat LLM — local (Ollama), Google AI Studio, or OpenCode Zen
    chat_provider: ChatProvider = "google"
    google_api_key: str = ""
    opencode_api_key: str = ""
    opencode_base_url: str = "https://opencode.ai/zen/go/v1"

    # Embeddings (local Ollama only)
    embedding_dimensions: int = 768
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"

    # Server
    allowed_origins: Annotated[list[str], NoDecode]

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

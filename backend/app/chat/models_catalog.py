"""Discover chat providers and list models from upstream APIs only."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from pydantic import BaseModel

from app.config import CHAT_PROVIDERS, ChatProvider, settings

GOOGLE_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class ModelCatalogError(RuntimeError):
    """Raised when a provider's live model catalog cannot be loaded."""


class ChatModelOption(BaseModel):
    id: str
    label: str


class ChatProviderCatalog(BaseModel):
    id: ChatProvider
    label: str
    default_model: str
    models: list[ChatModelOption]


class ChatProvidersResponse(BaseModel):
    default_provider: ChatProvider
    default_model: str
    providers: list[ChatProviderCatalog]


@dataclass(frozen=True)
class ResolvedChatModel:
    provider: ChatProvider
    model: str


def provider_label(provider: ChatProvider) -> str:
    labels = {
        "local": "Local (Ollama)",
        "google": "Google AI Studio",
        "opencode": "OpenCode Zen",
    }
    return labels[provider]


def configured_providers() -> list[ChatProvider]:
    providers: list[ChatProvider] = []
    if settings.use_ollama:
        providers.append("local")
    if settings.google_api_key.strip():
        providers.append("google")
    if settings.opencode_api_key.strip():
        providers.append("opencode")
    return providers


def resolve_chat_model(
    provider: ChatProvider | None,
    model: str | None,
) -> ResolvedChatModel:
    if provider is None:
        raise ValueError("provider is required")
    if provider not in CHAT_PROVIDERS:
        supported = ", ".join(sorted(CHAT_PROVIDERS))
        raise ValueError(f"provider must be one of: {supported}")
    if provider not in configured_providers():
        raise ValueError(f"Provider {provider!r} is not configured on this server")

    resolved_model = (model or "").strip()
    if not resolved_model:
        raise ValueError("model is required")

    allowed_ids = {option.id for option in list_models(provider)}
    if resolved_model not in allowed_ids:
        raise ValueError(
            f"Model {resolved_model!r} is not available for provider {provider!r}"
        )

    return ResolvedChatModel(provider=provider, model=resolved_model)


def list_models(provider: ChatProvider) -> list[ChatModelOption]:
    if provider == "local":
        return _fetch_ollama_models()
    if provider == "google":
        return _fetch_google_models()
    return _fetch_opencode_models()


def build_providers_response() -> ChatProvidersResponse:
    catalogs: list[ChatProviderCatalog] = []
    for provider in configured_providers():
        models = list_models(provider)
        if not models:
            continue
        catalogs.append(
            ChatProviderCatalog(
                id=provider,
                label=provider_label(provider),
                default_model=models[0].id,
                models=models,
            )
        )

    if not catalogs:
        raise ModelCatalogError("No chat models are available from configured providers")

    default_provider = settings.chat_provider
    default_catalog = next(
        (catalog for catalog in catalogs if catalog.id == default_provider),
        catalogs[0],
    )

    return ChatProvidersResponse(
        default_provider=default_catalog.id,
        default_model=default_catalog.default_model,
        providers=catalogs,
    )


def _fetch_ollama_models() -> list[ChatModelOption]:
    base = settings.ollama_base_url.rstrip("/")
    try:
        response = httpx.get(f"{base}/api/tags", timeout=10.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return []

    models: list[ChatModelOption] = []
    for entry in response.json().get("models", []):
        name = entry.get("name") or entry.get("model")
        if not name:
            continue
        capabilities = entry.get("capabilities") or []
        if capabilities and "completion" not in capabilities and "tools" not in capabilities:
            continue
        models.append(ChatModelOption(id=name, label=name))

    models.sort(key=lambda option: option.id)
    return models


def _fetch_google_models() -> list[ChatModelOption]:
    if not settings.google_api_key.strip():
        return []

    response = httpx.get(
        GOOGLE_MODELS_URL,
        params={"key": settings.google_api_key},
        timeout=10.0,
    )
    response.raise_for_status()

    models: list[ChatModelOption] = []
    for entry in response.json().get("models", []):
        name = entry.get("name", "")
        if not name.startswith("models/"):
            continue
        model_id = name.removeprefix("models/")
        methods = entry.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        display = entry.get("displayName") or model_id
        models.append(ChatModelOption(id=model_id, label=display))

    models.sort(key=lambda option: option.id)
    return models


def _fetch_opencode_models() -> list[ChatModelOption]:
    if not settings.opencode_api_key.strip():
        return []

    base = settings.opencode_base_url.rstrip("/")
    response = httpx.get(
        f"{base}/models",
        headers={"Authorization": f"Bearer {settings.opencode_api_key}"},
        timeout=10.0,
    )
    response.raise_for_status()

    models: list[ChatModelOption] = []
    for entry in response.json().get("data", []):
        model_id = entry.get("id")
        if not model_id:
            continue
        models.append(ChatModelOption(id=model_id, label=model_id))

    models.sort(key=lambda option: option.id)
    return models

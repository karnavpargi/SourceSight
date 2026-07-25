from unittest.mock import patch

import httpx
import pytest

from app.chat import models_catalog
from app.chat.models_catalog import (
    ChatModelOption,
    ModelCatalogError,
    ResolvedChatModel,
    build_providers_response,
    configured_providers,
    list_models,
    resolve_chat_model,
    resolve_router_model,
)
from app.config import settings


def test_resolve_chat_model_requires_model_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.chat.models_catalog.settings.chat_model", "")
    monkeypatch.setattr("app.chat.models_catalog.settings.chat_provider", "google")
    monkeypatch.setattr(
        "app.chat.models_catalog.configured_providers",
        lambda: ["google"],
    )

    with pytest.raises(ValueError, match="model is required"):
        resolve_chat_model("google", None)


def test_resolve_chat_model_uses_configured_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.chat.models_catalog.settings.chat_provider", "google")
    monkeypatch.setattr(
        "app.chat.models_catalog.settings.chat_model",
        "gemini-flash-lite-latest",
    )
    monkeypatch.setattr(
        "app.chat.models_catalog.configured_providers",
        lambda: ["google"],
    )

    resolved = resolve_chat_model(None, None)

    assert resolved == ResolvedChatModel(
        provider="google",
        model="gemini-flash-lite-latest",
    )


def test_resolve_chat_model_validates_against_live_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.chat.models_catalog.settings.chat_provider", "google")
    monkeypatch.setattr(
        "app.chat.models_catalog.settings.chat_model",
        "gemini-flash-lite-latest",
    )
    monkeypatch.setattr(
        "app.chat.models_catalog.configured_providers",
        lambda: ["google"],
    )
    monkeypatch.setattr(
        "app.chat.models_catalog.list_models",
        lambda provider: [ChatModelOption(id="gemini-2.0-flash", label="Gemini 2.0 Flash")],
    )

    resolved = resolve_chat_model("google", "gemini-2.0-flash")

    assert resolved == ResolvedChatModel(provider="google", model="gemini-2.0-flash")


def test_resolve_chat_model_rejects_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.chat.models_catalog.settings.chat_provider", "google")
    monkeypatch.setattr(
        "app.chat.models_catalog.settings.chat_model",
        "gemini-flash-lite-latest",
    )
    monkeypatch.setattr(
        "app.chat.models_catalog.configured_providers",
        lambda: ["google"],
    )
    monkeypatch.setattr(
        "app.chat.models_catalog.list_models",
        lambda provider: [ChatModelOption(id="gemini-2.0-flash", label="Gemini 2.0 Flash")],
    )

    with pytest.raises(ValueError, match="not available"):
        resolve_chat_model("google", "does-not-exist")


def test_list_opencode_models_uses_live_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.chat.models_catalog.settings.opencode_api_key", "test-key")
    monkeypatch.setattr(
        "app.chat.models_catalog.settings.opencode_base_url",
        "https://opencode.ai/zen/go/v1",
    )

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "data": [
                    {"id": "glm-5.2"},
                    {"id": "minimax-m3"},
                    {"id": "kimi-k2.7-code"},
                ]
            }

    with patch("app.chat.models_catalog.http_get", return_value=FakeResponse()):
        models = list_models("opencode")

    assert [model.id for model in models] == ["glm-5.2", "kimi-k2.7-code", "minimax-m3"]


def test_list_ollama_models_returns_empty_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with patch(
        "app.chat.models_catalog.http_get",
        side_effect=httpx.ConnectError("connection refused"),
    ):
        assert list_models("local") == []


def test_list_models_raises_when_live_api_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.chat.models_catalog.settings.opencode_api_key",
        "test-key",
    )

    with patch(
        "app.chat.models_catalog.http_get",
        side_effect=httpx.HTTPError("upstream unavailable"),
    ):
        with pytest.raises(ModelCatalogError, match="OpenCode Zen model catalog failed"):
            list_models("opencode")


def test_list_google_models_raises_catalog_error_on_http_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.chat.models_catalog.settings.google_api_key",
        "test-key",
    )

    with patch(
        "app.chat.models_catalog.http_get",
        side_effect=httpx.HTTPStatusError(
            "401",
            request=httpx.Request("GET", "https://example.test"),
            response=httpx.Response(401),
        ),
    ):
        with pytest.raises(ModelCatalogError, match="Google AI Studio model catalog failed"):
            list_models("google")


def test_list_google_models_excludes_retired_generate_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.chat.models_catalog.settings.google_api_key", "test-key")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "models": [
                    {
                        "name": "models/gemini-2.0-flash-lite",
                        "displayName": "Retired Flash Lite",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/gemini-flash-lite-latest",
                        "displayName": "Flash Lite Latest",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                ]
            }

    with patch("app.chat.models_catalog.http_get", return_value=FakeResponse()):
        models = list_models("google")

    assert [model.id for model in models] == ["gemini-flash-lite-latest"]


def test_list_google_models_excludes_non_tool_specializations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.chat.models_catalog.settings.google_api_key", "test-key")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "models": [
                    {
                        "name": "models/antigravity-preview-05-2026",
                        "displayName": "Antigravity",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/gemini-2.5-flash-image",
                        "displayName": "Image",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/gemini-flash-latest",
                        "displayName": "Gemini Flash Latest",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/gemma-4-31b-it",
                        "displayName": "Gemma 4",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                ]
            }

    with patch("app.chat.models_catalog.http_get", return_value=FakeResponse()):
        models = list_models("google")

    assert [model.id for model in models] == ["gemini-flash-latest", "gemma-4-31b-it"]


def test_build_providers_response_prefers_configured_chat_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.chat.models_catalog.configured_providers",
        lambda: ["google"],
    )
    monkeypatch.setattr(
        "app.chat.models_catalog.list_models",
        lambda provider: [
            ChatModelOption(id="gemma-4-31b-it", label="Gemma"),
            ChatModelOption(id="gemini-3-flash-preview", label="Gemini 3 Flash Preview"),
            ChatModelOption(id="gemini-flash-latest", label="Gemini Flash Latest"),
        ],
    )
    monkeypatch.setattr("app.chat.models_catalog.settings.chat_provider", "google")
    monkeypatch.setattr(
        "app.chat.models_catalog.settings.chat_model",
        "gemini-flash-lite-latest",
    )

    response = build_providers_response()

    assert response.default_provider == "google"
    assert response.default_model == "gemini-flash-lite-latest"
    assert response.providers[0].models[0].id == "gemini-flash-lite-latest"


def test_build_providers_response_uses_live_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.chat.models_catalog.configured_providers",
        lambda: ["local", "google", "opencode"],
    )
    monkeypatch.setattr(
        "app.chat.models_catalog.list_models",
        lambda provider: [ChatModelOption(id=f"{provider}-live", label=f"{provider}-live")],
    )
    monkeypatch.setattr("app.chat.models_catalog.settings.chat_provider", "opencode")
    monkeypatch.setattr("app.chat.models_catalog.settings.chat_model", "opencode-live")

    response = build_providers_response()

    assert response.default_provider == "opencode"
    assert response.default_model == "opencode-live"
    assert response.providers[0].default_model == "local-live"


def test_build_providers_response_fails_when_no_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.chat.models_catalog.configured_providers",
        lambda: ["local"],
    )
    monkeypatch.setattr("app.chat.models_catalog.list_models", lambda provider: [])

    with pytest.raises(ModelCatalogError, match="No chat models are available"):
        build_providers_response()


def test_resolve_router_model_returns_configured_live_google_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "chat_router_model", "gemini-2.0-flash-lite")
    monkeypatch.setattr(settings, "google_api_key", "key")
    monkeypatch.setattr(
        models_catalog,
        "_cached_google_model_ids",
        lambda: frozenset({"gemini-2.0-flash-lite"}),
    )
    assert resolve_router_model() == ResolvedChatModel(
        provider="google",
        model="gemini-2.0-flash-lite",
    )


def test_resolve_router_model_returns_none_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "chat_router_model", "retired-model")
    monkeypatch.setattr(settings, "google_api_key", "key")
    monkeypatch.setattr(
        models_catalog,
        "_cached_google_model_ids",
        lambda: frozenset({"gemini-3.5-flash-lite"}),
    )
    assert resolve_router_model() is None


def test_configured_providers_requires_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.chat.models_catalog.settings.use_ollama", True)
    monkeypatch.setattr("app.chat.models_catalog.settings.google_api_key", "")
    monkeypatch.setattr("app.chat.models_catalog.settings.opencode_api_key", "")

    assert configured_providers() == ["local"]


def test_configured_providers_omits_local_when_ollama_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.chat.models_catalog.settings.use_ollama", False)
    monkeypatch.setattr("app.chat.models_catalog.settings.google_api_key", "")
    monkeypatch.setattr("app.chat.models_catalog.settings.opencode_api_key", "test-key")

    assert configured_providers() == ["opencode"]

from unittest.mock import patch

import httpx
import pytest

from app.chat.models_catalog import (
    ChatModelOption,
    ModelCatalogError,
    ResolvedChatModel,
    build_providers_response,
    configured_providers,
    list_models,
    resolve_chat_model,
)


def test_resolve_chat_model_requires_provider_and_model() -> None:
    with pytest.raises(ValueError, match="provider is required"):
        resolve_chat_model(None, "gemini-2.0-flash")

    with pytest.raises(ValueError, match="model is required"):
        resolve_chat_model("google", None)


def test_resolve_chat_model_validates_against_live_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.chat.models_catalog.list_models",
        lambda provider: [ChatModelOption(id="gemini-2.0-flash", label="Gemini 2.0 Flash")],
    )

    resolved = resolve_chat_model("google", "gemini-2.0-flash")

    assert resolved == ResolvedChatModel(provider="google", model="gemini-2.0-flash")


def test_resolve_chat_model_rejects_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
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

    with patch("app.chat.models_catalog.httpx.get", return_value=FakeResponse()):
        models = list_models("opencode")

    assert [model.id for model in models] == ["glm-5.2", "kimi-k2.7-code", "minimax-m3"]


def test_list_models_raises_when_live_api_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.chat.models_catalog.settings.opencode_api_key",
        "test-key",
    )

    with patch(
        "app.chat.models_catalog.httpx.get",
        side_effect=httpx.HTTPError("upstream unavailable"),
    ):
        with pytest.raises(httpx.HTTPError):
            list_models("opencode")


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


def test_configured_providers_requires_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.chat.models_catalog.settings.google_api_key", "")
    monkeypatch.setattr("app.chat.models_catalog.settings.opencode_api_key", "")

    assert configured_providers() == ["local"]

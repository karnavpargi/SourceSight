from app.chat.generation import ChatGenerationConfig, build_model_settings


def test_build_model_settings_uses_temperature() -> None:
    settings = build_model_settings(ChatGenerationConfig(), max_tokens=1000)
    assert settings["temperature"] == 1.0

    settings = build_model_settings(
        ChatGenerationConfig(temperature=0.5),
        max_tokens=500,
    )
    assert settings["temperature"] == 0.5


def test_build_model_settings_includes_max_tokens() -> None:
    settings = build_model_settings(ChatGenerationConfig(), max_tokens=1500)
    assert settings["temperature"] == 1.0
    assert settings["max_tokens"] == 1500

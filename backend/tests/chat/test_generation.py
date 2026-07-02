from app.chat.generation import ChatGenerationConfig, build_model_settings


def test_build_model_settings_uses_temperature() -> None:
    assert build_model_settings(ChatGenerationConfig()) == {"temperature": 1.0}
    assert build_model_settings(ChatGenerationConfig(temperature=0.5)) == {
        "temperature": 0.5
    }

"""Per-request LLM generation settings from the chat UI."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.settings import ModelSettings


@dataclass(frozen=True)
class ChatGenerationConfig:
    temperature: float = 1.0


def build_model_settings(config: ChatGenerationConfig) -> ModelSettings:
    return {"temperature": config.temperature}

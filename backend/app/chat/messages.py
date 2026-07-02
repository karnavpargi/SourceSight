"""AI SDK wire-format helpers."""

from __future__ import annotations


def extract_latest_user_text(messages: list[dict]) -> str:
    """Return the text of the most recent user message from AI SDK UI messages."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue

        parts = message.get("parts")
        if isinstance(parts, list):
            text_parts = [
                part["text"]
                for part in parts
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
            ]
            if text_parts:
                return "".join(text_parts).strip()

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    raise ValueError("No user message found in request.")

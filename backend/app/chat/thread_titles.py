"""Helpers for deriving chat thread titles from user prompts."""

from __future__ import annotations

DEFAULT_THREAD_TITLE = "New analysis"
MAX_THREAD_TITLE_LENGTH = 255


def derive_thread_title(user_text: str) -> str:
    """Build a short sidebar title from the first user message."""
    normalized = " ".join(user_text.split()).strip()
    if not normalized:
        return DEFAULT_THREAD_TITLE

    if len(normalized) <= MAX_THREAD_TITLE_LENGTH:
        return normalized

    truncated = normalized[:MAX_THREAD_TITLE_LENGTH]
    last_space = truncated.rfind(" ")
    if last_space > MAX_THREAD_TITLE_LENGTH // 2:
        truncated = truncated[:last_space]

    return truncated.rstrip(".,;:!? ") + "…"

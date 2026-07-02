"""Per-request LLM generation settings from the chat UI."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic_ai.settings import ModelSettings

from app.assistant.outputs import GroundedAnswer

DEFAULT_MAX_OUTPUT_TOKENS = 300
STRUCTURED_OUTPUT_TOKEN_OVERHEAD = 512
_CITATION_MARKER = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class ChatGenerationConfig:
    temperature: float = 1.0
    max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS


def approximate_word_count(max_output_tokens: int) -> int:
    return max(1, int(max_output_tokens * 0.75))


def estimate_token_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // 4)


def build_length_instruction(max_output_tokens: int) -> str:
    words = approximate_word_count(max_output_tokens)
    return (
        f"\n\nResponse length limit: keep the `answer` field under "
        f"{max_output_tokens} tokens (about {words} words). "
        "Use the shortest format that still answers the question. "
        "Prefer one compact list over multiple sections, and skip closing summaries."
    )


def build_model_settings(config: ChatGenerationConfig) -> ModelSettings:
    settings: ModelSettings = {"temperature": config.temperature}
    if config.max_output_tokens is not None:
        settings["max_tokens"] = config.max_output_tokens + STRUCTURED_OUTPUT_TOKEN_OVERHEAD
    return settings


def truncate_answer_text(text: str, max_output_tokens: int) -> str:
    if estimate_token_count(text) <= max_output_tokens:
        return text

    max_chars = max_output_tokens * 4
    truncated = text[:max_chars].rstrip()
    for separator in (". ", ".\n", "\n"):
        boundary = truncated.rfind(separator)
        if boundary >= max_chars // 2:
            return truncated[: boundary + 1].rstrip()

    return f"{truncated.rstrip('.,;:!?')}…"


def limit_grounded_answer(
    answer: GroundedAnswer,
    max_output_tokens: int | None,
) -> GroundedAnswer:
    if max_output_tokens is None:
        return answer

    limited_answer = truncate_answer_text(answer.answer, max_output_tokens)
    if limited_answer == answer.answer:
        return answer

    cited_indices = {
        int(match.group(1)) for match in _CITATION_MARKER.finditer(limited_answer)
    }
    citations = [
        citation
        for citation in answer.citations
        if citation.citation_index in cited_indices
    ]
    cited_chunk_ids = {citation.chunk_id for citation in citations}
    cited_passages = [
        passage
        for passage in answer.cited_passages
        if passage.chunk_id in cited_chunk_ids
    ]

    return GroundedAnswer(
        answer=limited_answer,
        citations=citations,
        cited_passages=cited_passages,
    )

from app.assistant.outputs import Citation, GroundedAnswer
from app.chat.generation import (
    ChatGenerationConfig,
    DEFAULT_MAX_OUTPUT_TOKENS,
    STRUCTURED_OUTPUT_TOKEN_OVERHEAD,
    approximate_word_count,
    build_length_instruction,
    build_model_settings,
    estimate_token_count,
    limit_grounded_answer,
    truncate_answer_text,
)
from uuid import UUID

CHUNK_ID = UUID("11111111-1111-1111-1111-111111111111")


def test_build_model_settings_default_includes_structured_overhead() -> None:
    settings = build_model_settings(ChatGenerationConfig())
    assert settings == {
        "temperature": 1.0,
        "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS + STRUCTURED_OUTPUT_TOKEN_OVERHEAD,
    }


def test_build_model_settings_omits_max_tokens_when_unlimited() -> None:
    settings = build_model_settings(
        ChatGenerationConfig(temperature=0.5, max_output_tokens=None)
    )
    assert settings == {"temperature": 0.5}


def test_build_length_instruction_mentions_token_and_word_limits() -> None:
    instruction = build_length_instruction(300)
    assert "300 tokens" in instruction
    assert f"{approximate_word_count(300)} words" in instruction


def test_estimate_token_count_uses_character_heuristic() -> None:
    assert estimate_token_count("abcd") == 1
    assert estimate_token_count("a" * 40) == 10


def test_truncate_answer_text_shortens_long_prose() -> None:
    long_answer = "Word. " * 200
    truncated = truncate_answer_text(long_answer, max_output_tokens=40)
    assert estimate_token_count(truncated) <= 40


def test_limit_grounded_answer_drops_citations_removed_by_truncation() -> None:
    answer = GroundedAnswer(
        answer="First claim [1]. Second claim [2]. Third claim [3].",
        citations=[
            Citation(citation_index=1, chunk_id=CHUNK_ID, excerpt="one"),
            Citation(citation_index=2, chunk_id=CHUNK_ID, excerpt="two"),
            Citation(citation_index=3, chunk_id=CHUNK_ID, excerpt="three"),
        ],
        cited_passages=[],
    )

    limited = limit_grounded_answer(answer, max_output_tokens=8)

    assert "[3]" not in limited.answer
    assert "[2]" not in limited.answer
    assert [citation.citation_index for citation in limited.citations] == [1]

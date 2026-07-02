from __future__ import annotations

from ingest.chunk import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    chunk_markdown,
    estimate_tokens,
)


def test_chunk_splits_on_section_headers() -> None:
    markdown = "\n".join(
        [
            "Cover page preamble.",
            "## Item 1. Business",
            "Business paragraph one.",
            "## Item 1A. Risk Factors",
            "Risk paragraph one.",
        ]
    )
    chunks = chunk_markdown(markdown, target_tokens=50, overlap_tokens=10)

    sections = {chunk.section for chunk in chunks}
    assert "Item 1. Business" in sections
    assert "Item 1A. Risk Factors" in sections
    assert all(chunk.chunk_index == index for index, chunk in enumerate(chunks))


def test_chunk_overlap_is_shared_between_neighbors() -> None:
    paragraph = "Supply chain concentration risk. " * 120
    markdown = f"## Item 1A. Risk Factors\n\n{paragraph}"
    chunks = chunk_markdown(
        markdown,
        target_tokens=DEFAULT_TARGET_TOKENS,
        overlap_tokens=DEFAULT_OVERLAP_TOKENS,
    )
    assert len(chunks) >= 2

    for previous, current in zip(chunks, chunks[1:], strict=False):
        overlap = _shared_suffix_prefix(previous.content, current.content)
        assert estimate_tokens(overlap) >= DEFAULT_OVERLAP_TOKENS // 2


def test_chunk_respects_target_size_for_small_sections() -> None:
    markdown = "## Item 2. Properties\n\nShort section body."
    chunks = chunk_markdown(markdown)
    assert len(chunks) == 1
    assert chunks[0].section == "Item 2. Properties"
    assert chunks[0].token_count <= DEFAULT_TARGET_TOKENS


def test_chunk_indexes_are_sequential() -> None:
    body = "\n\n".join(f"Paragraph {index}. " + ("word " * 200) for index in range(8))
    markdown = f"## Item 7. Management s Discussion and Analysis\n\n{body}"
    chunks = chunk_markdown(markdown)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunk_skips_empty_section_body() -> None:
    markdown = "## Item 1. Business\n\n   \n## Item 1A. Risk Factors\n\nActual content."
    chunks = chunk_markdown(markdown)
    assert len(chunks) == 1
    assert chunks[0].section == "Item 1A. Risk Factors"


def test_chunk_splits_oversized_paragraph_with_overlap() -> None:
    long_sentence = "Risk sentence with detail. " * 400
    markdown = f"## Item 1A. Risk Factors\n\n{long_sentence}"
    chunks = chunk_markdown(markdown, target_tokens=100, overlap_tokens=20)
    assert len(chunks) >= 2


def test_chunk_tail_overlap_uses_full_text_when_short() -> None:
    markdown = "## Item 1. Business\n\n" + ("word " * 30)
    chunks = chunk_markdown(markdown, target_tokens=10, overlap_tokens=50)
    assert chunks


def test_chunk_flushes_pending_parts_before_long_paragraph() -> None:
    prefix = "Short intro. "
    long_paragraph = "Sentence. " * 500
    markdown = f"## Item 1A. Risk Factors\n\n{prefix}\n\n{long_paragraph}"
    chunks = chunk_markdown(markdown, target_tokens=80, overlap_tokens=10)
    assert len(chunks) >= 2
    assert "Short intro." in chunks[0].content


def test_estimate_tokens_minimum_one() -> None:
    assert estimate_tokens("") == 1


def test_chunk_section_with_no_paragraphs_returns_empty() -> None:
    from ingest.chunk import _chunk_section

    body = ("   \n\n" * 300)
    chunks = _chunk_section(body, section="Item 1. Business", target_tokens=10, overlap_tokens=2)
    assert chunks == []


def test_chunk_split_long_paragraph_skips_blank_sentences() -> None:
    from unittest.mock import patch

    from ingest.chunk import _split_long_paragraph

    with patch("ingest.chunk.re.split", return_value=["", "Actual sentence here."]):
        pieces = _split_long_paragraph("ignored", target_tokens=5)
    assert pieces == ["Actual sentence here."]


def test_chunk_tail_overlap_without_paragraph_break() -> None:
    from ingest.chunk import _tail_for_overlap

    text = "z" * 300
    tail = _tail_for_overlap(text, overlap_tokens=10)
    assert tail == text[-40:]


def test_chunk_tail_overlap_uses_paragraph_break() -> None:
    from ingest.chunk import _tail_for_overlap

    text = ("x" * 120) + "\n\n" + ("y" * 10)
    tail = _tail_for_overlap(text, overlap_tokens=20)
    assert tail.startswith("y")


def _shared_suffix_prefix(left: str, right: str) -> str:
    max_length = min(len(left), len(right))
    for size in range(max_length, 0, -1):
        if left.endswith(right[:size]):
            return right[:size]
    return ""

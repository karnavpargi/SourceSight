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


def _shared_suffix_prefix(left: str, right: str) -> str:
    max_length = min(len(left), len(right))
    for size in range(max_length, 0, -1):
        if left.endswith(right[:size]):
            return right[:size]
    return ""

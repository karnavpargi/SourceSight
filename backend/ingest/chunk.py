"""Section-aware Markdown chunker tuned for SEC 10-K filings."""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_TARGET_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 150
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    content: str
    section: str | None
    page: int | None
    token_count: int


def chunk_markdown(
    markdown: str,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[TextChunk]:
    """Split Markdown into retrieval-ready chunks respecting 10-K section headers."""
    chunks: list[TextChunk] = []
    for section, body in _split_sections(markdown):
        if not body.strip():
            continue
        chunks.extend(
            _chunk_section(
                body,
                section=section,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            )
        )

    return [
        TextChunk(
            chunk_index=index,
            content=chunk.content,
            section=chunk.section,
            page=chunk.page,
            token_count=chunk.token_count,
        )
        for index, chunk in enumerate(chunks)
    ]


def estimate_tokens(text: str) -> int:
    """Approximate OpenAI token count without pulling in tiktoken."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_sections(markdown: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    current_section: str | None = None
    current_lines: list[str] = []

    for line in markdown.splitlines():
        if line.startswith("## "):
            if current_lines or current_section is not None:
                sections.append((current_section, "\n".join(current_lines).strip()))
            current_section = line[3:].strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines or current_section is not None:
        sections.append((current_section, "\n".join(current_lines).strip()))

    return sections


def _split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]


def _chunk_section(
    body: str,
    *,
    section: str | None,
    target_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    if estimate_tokens(body) <= target_tokens:
        return [_make_chunk(body, section=section)]

    paragraphs = _split_paragraphs(body)
    if not paragraphs:
        return []

    chunks: list[TextChunk] = []
    current_parts: list[str] = []
    current_tokens = 0
    overlap_prefix = ""

    for paragraph in paragraphs:
        paragraph_tokens = estimate_tokens(paragraph)
        if paragraph_tokens > target_tokens:
            if current_parts:
                chunk_text = _join_parts(current_parts)
                chunks.append(_make_chunk(chunk_text, section=section))
                overlap_prefix = _tail_for_overlap(chunk_text, overlap_tokens)
                current_parts = []
                current_tokens = 0

            for piece in _split_long_paragraph(paragraph, target_tokens):
                if overlap_prefix:
                    piece = f"{overlap_prefix}\n\n{piece}"
                    overlap_prefix = ""
                chunks.append(_make_chunk(piece, section=section))
                overlap_prefix = _tail_for_overlap(piece, overlap_tokens)
            continue

        next_tokens = current_tokens + paragraph_tokens
        if current_parts and next_tokens > target_tokens:
            chunk_text = _join_parts(current_parts)
            chunks.append(_make_chunk(chunk_text, section=section))
            overlap_prefix = _tail_for_overlap(chunk_text, overlap_tokens)
            current_parts = [overlap_prefix, paragraph] if overlap_prefix else [paragraph]
            current_tokens = estimate_tokens(_join_parts(current_parts))
            overlap_prefix = ""
            continue

        current_parts.append(paragraph)
        current_tokens = next_tokens

    if current_parts:
        chunk_text = _join_parts(current_parts)
        chunks.append(_make_chunk(chunk_text, section=section))

    return chunks


def _split_long_paragraph(paragraph: str, target_tokens: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = estimate_tokens(sentence)
        if current and current_tokens + sentence_tokens > target_tokens:
            pieces.append(" ".join(current))
            current = [sentence]
            current_tokens = sentence_tokens
            continue
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        pieces.append(" ".join(current))
    return pieces


def _tail_for_overlap(text: str, overlap_tokens: int) -> str:
    char_target = overlap_tokens * _CHARS_PER_TOKEN
    if len(text) <= char_target:
        return text

    tail = text[-char_target:]
    paragraph_break = tail.find("\n\n")
    if paragraph_break != -1:
        return tail[paragraph_break + 2 :]
    return tail


def _join_parts(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part.strip())


def _make_chunk(content: str, *, section: str | None) -> TextChunk:
    normalized = content.strip()
    return TextChunk(
        chunk_index=-1,
        content=normalized,
        section=section,
        page=None,
        token_count=estimate_tokens(normalized),
    )

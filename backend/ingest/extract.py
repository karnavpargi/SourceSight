"""SEC filing HTML → clean Markdown for ingestion."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path

# 10-K section headings and cover-page labels worth promoting to Markdown headers.
_HEADING_LINE_RE = re.compile(
    r"^(?:"
    r"PART\s+[IVXLC]+"
    r"|Item\s+\d+[A-Z]?\.\s*.+"
    r"|Financial Statement Line Items"
    r"|Documents filed as part of this report"
    r")$",
    re.IGNORECASE,
)

_TOC_START_RE = re.compile(r"Table\s+of\s+Contents", re.IGNORECASE)

# First real body section after a TOC block (company-specific anchor markers).
_TOC_END_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"Item\s+1\.(?:&#160;|\s)+Business</span></div><div\s+id=",
        re.IGNORECASE,
    ),
    re.compile(r"PART\s+I</span></div><div\s+id=", re.IGNORECASE),
    re.compile(r"Item\s+1\.</span></div><div\s+id=", re.IGNORECASE),
    re.compile(
        r"Item\s+1\.(?:&#160;|\s)+Business</span></div>\s*<div\s+id=",
        re.IGNORECASE,
    ),
)

_HIDDEN_DIV_START_RE = re.compile(
    r'<div\b[^>]*style="[^"]*display\s*:\s*none[^"]*"[^>]*>',
    re.IGNORECASE,
)

_XBRL_TAG_RE = re.compile(
    r"</?(?:ix|xbrli|link|xbrldi):[a-zA-Z][^>]*>",
    re.IGNORECASE,
)

_EMBEDDED_BLOCK_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def extract_markdown(html_content: str) -> str:
    """Convert SEC 10-K HTML to clean Markdown."""
    parser = _HtmlToMarkdown()
    parser.feed(_preprocess_html(html_content))
    return _postprocess_markdown("".join(parser.parts))


def extract_markdown_from_path(path: Path | str) -> str:
    """Read a local SEC HTML filing and return Markdown."""
    return extract_markdown(_read_html(path))


def _read_html(path: Path | str) -> str:
    filing_path = Path(path)
    return filing_path.read_text(encoding="utf-8", errors="replace")


def _preprocess_html(raw_html: str) -> str:
    cleaned = _HTML_COMMENT_RE.sub("", raw_html)
    cleaned = _remove_hidden_blocks(cleaned)
    cleaned = _XBRL_TAG_RE.sub("", cleaned)
    cleaned = _EMBEDDED_BLOCK_RE.sub("", cleaned)
    cleaned = _remove_table_of_contents(cleaned)
    cleaned = _strip_internal_links(cleaned)
    return cleaned


def _remove_hidden_blocks(text: str) -> str:
    """Drop inline XBRL metadata blocks (display:none)."""
    result: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        match = _HIDDEN_DIV_START_RE.search(text, index)
        if not match:
            result.append(text[index:])
            break

        result.append(text[index : match.start()])
        position = match.end()
        depth = 1

        while position < length and depth > 0:
            next_open = text.find("<div", position)
            next_close = text.find("</div>", position)
            if next_close == -1:
                position = length
                break
            if next_open != -1 and next_open < next_close:
                depth += 1
                position = next_open + 4
            else:
                depth -= 1
                position = next_close + 6

        index = position

    return "".join(result)


def _remove_table_of_contents(text: str) -> str:
    match = _TOC_START_RE.search(text)
    if not match:
        return text

    after_toc = text[match.end() :]
    for pattern in _TOC_END_PATTERNS:
        end_match = pattern.search(after_toc)
        if end_match:
            return text[: match.start()] + after_toc[end_match.start() :]

    return text[: match.start()] + after_toc


def _strip_internal_links(text: str) -> str:
    """Remove in-document anchor navigation; keep link text."""
    return re.sub(
        r'<a\b[^>]*href="#[^"]*"[^>]*>(.*?)</a>',
        r"\1",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )


def _normalize_inline_text(text: str) -> str:
    decoded = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", decoded).strip()


def _is_bold(attrs: dict[str, str]) -> bool:
    style = attrs.get("style", "").replace(" ", "").lower()
    return "font-weight:700" in style or "font-weight:bold" in style


class _HtmlToMarkdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._bold_depth = 0
        self._in_table = False
        self._in_cell = False
        self._cell_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip_depth:
            self._skip_depth += 1
            return

        if tag in ("script", "style", "meta", "link", "head"):
            self._skip_depth = 1
            return

        attrs_dict = {key: value or "" for key, value in attrs}
        if tag in ("span", "div", "p", "b", "strong", "td", "th") and _is_bold(attrs_dict):
            self._bold_depth += 1

        if tag == "table":
            self._in_table = True
            self.parts.append("\n\n")
        elif tag == "tr" and self._in_table:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_parts = []
        elif tag == "br":
            self.parts.append("\n")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            self.parts.append("\n\n" + ("#" * level) + " ")

    def handle_endtag(self, tag: str) -> None:
        if self._skip_depth:
            self._skip_depth -= 1
            return

        if tag in ("span", "div", "p", "b", "strong", "td", "th") and self._bold_depth:
            self._bold_depth -= 1

        if tag in ("td", "th") and self._in_cell:
            cell_text = _normalize_inline_text("".join(self._cell_parts))
            if cell_text:
                self.parts.append(cell_text + " | ")
            self._in_cell = False
            self._cell_parts = []
        elif tag == "table":
            self._in_table = False
            self.parts.append("\n\n")
        elif tag in ("div", "p") and not self._in_table:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return

        if self._in_cell:
            self._cell_parts.append(data)
            return

        normalized = _normalize_inline_text(data)
        if not normalized:
            return

        if self._bold_depth and _HEADING_LINE_RE.match(normalized):
            self.parts.append(f"\n\n## {normalized}\n\n")
        else:
            self.parts.append(normalized + " ")


def _postprocess_markdown(markdown: str) -> str:
    lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if line.startswith("## "):
            lines.append(line)
            continue
        if _HEADING_LINE_RE.match(line):
            lines.append(f"## {line}")
        else:
            lines.append(line)

    collapsed = "\n".join(lines)
    collapsed = re.sub(r"[ \t]+\n", "\n", collapsed)
    collapsed = re.sub(r"\n[ \t]+", "\n", collapsed)
    collapsed = re.sub(r" +", " ", collapsed)
    collapsed = re.sub(r" *\| *\n", "\n", collapsed)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2 or sys.argv[1].startswith("-"):
        raise SystemExit("Usage: uv run python -m ingest.extract <path-to-filing.htm>")

    output = extract_markdown_from_path(sys.argv[1])
    print(output[:4000])
    print(f"\n--- ({len(output):,} characters total) ---")

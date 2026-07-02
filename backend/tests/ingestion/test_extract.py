from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ingest.extract import extract_markdown, extract_markdown_from_path


def test_extract_strips_xbrl_tags() -> None:
    html = """
    <html><body>
    <ix:nonNumeric>hidden fact</ix:nonNumeric>
    <p>Revenue grew in fiscal 2024.</p>
  </body></html>
    """
    markdown = extract_markdown(html)
    assert "ix:nonNumeric" not in markdown
    assert "Revenue grew in fiscal 2024." in markdown


def test_extract_removes_hidden_display_none_blocks() -> None:
    html = """
    <html><body>
    <div style="display:none">XBRL metadata block</div>
    <p>Item 1. Business overview text.</p>
    </body></html>
    """
    markdown = extract_markdown(html)
    assert "XBRL metadata block" not in markdown
    assert "Business overview text." in markdown


def test_extract_strips_table_of_contents() -> None:
    html = """
    <html><body>
    <div>Table of Contents</div>
    <div>Item 1. Business ........ 5</div>
    <div>Item 1A. Risk Factors ........ 10</div>
    <div id="item1"><span style="font-weight:700">Item 1.&#160;Business</span></div><div id="body">
    <p>Apple designs consumer electronics.</p>
    </div>
    </body></html>
    """
    markdown = extract_markdown(html)
    assert "Table of Contents" not in markdown
    assert "Risk Factors ........ 10" not in markdown
    assert "Apple designs consumer electronics." in markdown


def test_extract_promotes_item_headings_to_markdown_headers() -> None:
    html = """
    <html><body>
    <div><span style="font-weight:700">Item 1A. Risk Factors</span></div>
    <p>Supply chain concentration remains a risk.</p>
    </body></html>
    """
    markdown = extract_markdown(html)
    assert "## Item 1A. Risk Factors" in markdown
    assert "Supply chain concentration remains a risk." in markdown


def test_extract_strips_internal_anchor_links() -> None:
    html = """
    <html><body>
    <a href="#item1a">Item 1A. Risk Factors</a>
    <p>See risks above.</p>
    </body></html>
    """
    markdown = extract_markdown(html)
    assert 'href="#' not in markdown
    assert "Item 1A. Risk Factors" in markdown


def test_extract_from_path(tmp_path: Path) -> None:
    filing = tmp_path / "filing.htm"
    filing.write_text("<html><body><p>Filing body text.</p></body></html>", encoding="utf-8")
    markdown = extract_markdown_from_path(filing)
    assert "Filing body text." in markdown


def test_extract_handles_tables_headings_and_skipped_tags() -> None:
    html = """
    <html><body>
    <script>ignore()</script>
    <h1>Cover Title</h1>
    <table><tr><th>Metric</th><td>Revenue</td></tr></table>
    <div><span style="font-weight:bold">PART I</span></div>
    <p>Regular paragraph.</p>
    </body></html>
    """
    markdown = extract_markdown(html)
    assert "ignore()" not in markdown
    assert "Metric" in markdown
    assert "Revenue" in markdown
    assert "## PART I" in markdown
    assert "Regular paragraph." in markdown


def test_extract_toc_fallback_when_end_marker_missing() -> None:
    html = """
    <html><body>
    <div>Table of Contents</div>
    <div>Item 1. Business ........ 5</div>
    <p>Content after TOC with no end marker.</p>
    </body></html>
    """
    markdown = extract_markdown(html)
    assert "Table of Contents" not in markdown
    assert "Content after TOC with no end marker." in markdown


def test_extract_skips_script_content_in_nested_tags() -> None:
    html = """
    <html><body>
    <div><style>.hidden { display:none; }</style><p>After style block.</p></div>
    <br/>
    <p>Line break above.</p>
    </body></html>
    """
    markdown = extract_markdown(html)
    assert ".hidden" not in markdown
    assert "After style block." in markdown
    assert "Line break above." in markdown

    html = """
    <html><body>
    <div style="display:none">hidden
    <div>nested hidden</div>
    </div>
    <p>Visible paragraph.</p>
    </body></html>
    """
    markdown = extract_markdown(html)
    assert "hidden" not in markdown
    assert "Visible paragraph." in markdown


def test_extract_main_usage_error() -> None:
    with pytest.raises(SystemExit, match="Usage"):
        import runpy

        runpy.run_module("ingest.extract", run_name="__main__")


def test_extract_main_with_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    filing = tmp_path / "filing.htm"
    filing.write_text("<html><body><p>Main entry text.</p></body></html>", encoding="utf-8")
    import runpy
    import sys

    with patch.object(sys, "argv", ["extract", str(filing)]):
        runpy.run_module("ingest.extract", run_name="__main__")
    output = capsys.readouterr().out
    assert "Main entry text." in output


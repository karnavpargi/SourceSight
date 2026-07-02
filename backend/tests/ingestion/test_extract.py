from __future__ import annotations

import pytest

from ingest.extract import extract_markdown


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

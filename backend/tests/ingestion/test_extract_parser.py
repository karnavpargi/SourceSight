from __future__ import annotations

from ingest.extract import _HtmlToMarkdown, _remove_hidden_blocks


def test_remove_hidden_blocks_handles_nested_inner_div() -> None:
    html = """
    <div style="display:none">outer
    <div>inner still hidden</div>
    tail hidden</div>
    <p>Visible paragraph.</p>
    """
    cleaned = _remove_hidden_blocks(html)
    assert "inner still hidden" not in cleaned
    assert "Visible paragraph." in cleaned


def test_html_parser_ignores_data_inside_skipped_tags() -> None:
    parser = _HtmlToMarkdown()
    parser.handle_starttag("script", [])
    parser.handle_data("ignored script body")
    parser.handle_endtag("script")
    parser.handle_data("Visible.")
    assert "ignored script body" not in "".join(parser.parts)
    assert "Visible." in "".join(parser.parts)

    parser = _HtmlToMarkdown()
    parser.handle_starttag("script", [])
    parser.handle_starttag("div", [])
    parser.handle_endtag("div")
    parser.handle_endtag("script")
    parser.feed("<p>Visible.</p>")
    parser.close()
    assert "Visible." in "".join(parser.parts)


def test_remove_hidden_blocks_without_closing_tag() -> None:
    html = '<div style="display:none">never closed'
    cleaned = _remove_hidden_blocks(html)
    assert "never closed" not in cleaned


def test_html_parser_inserts_line_breaks_and_headings() -> None:
    parser = _HtmlToMarkdown()
    parser.feed(
        "<h2>Section</h2><br><div><span style='font-weight:bold'>"
        "Item 1. Business</span></div><p>Body.</p>"
    )
    parser.close()
    output = "".join(parser.parts)
    assert "## Section" in output or "Section" in output
    assert "Item 1. Business" in output
    assert "Body." in output

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingest.chunk import TextChunk
from ingest.run import FilingRecord, IngestStats, ingest_filing, load_manifest, main, run_ingest


def test_filing_record_from_manifest_and_properties() -> None:
    record = FilingRecord.from_manifest_entry(
        {
            "ticker": "AAPL",
            "cik": "0000320193",
            "form": "10-K",
            "filing_date": "2024-11-01",
            "report_date": "2024-09-28",
            "accession_number": "0000320193-24-000123",
            "primary_document": "aapl.htm",
            "source_url": "https://example.com",
            "local_path": "2024/aapl.htm",
        }
    )
    assert record.fiscal_year == 2024
    assert record.company_name == "Apple Inc."


def test_filing_record_without_report_date_uses_filing_date() -> None:
    record = FilingRecord.from_manifest_entry(
        {
            "ticker": "AAPL",
            "cik": "0000320193",
            "form": "10-K",
            "filing_date": "2024-11-01",
            "report_date": "",
            "accession_number": "0000320193-24-000123",
            "primary_document": "aapl.htm",
            "source_url": "https://example.com",
            "local_path": "2024/aapl.htm",
        }
    )
    assert record.report_date is None
    assert record.fiscal_year == 2024


def test_load_manifest(tmp_path: Path) -> None:
    manifest = {
        "filings": [
            {
                "ticker": "AAPL",
                "cik": "0000320193",
                "form": "10-K",
                "filing_date": "2024-11-01",
                "report_date": "2024-09-28",
                "accession_number": "0000320193-24-000123",
                "primary_document": "aapl.htm",
                "source_url": "https://example.com",
                "local_path": "2024/aapl.htm",
            }
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    filings = load_manifest(path)
    assert len(filings) == 1
    assert filings[0].ticker == "AAPL"


def test_ingest_filing_skips_existing() -> None:
    filing = FilingRecord(
        ticker="AAPL",
        cik="0000320193",
        form_type="10-K",
        filing_date=date(2024, 11, 1),
        report_date=date(2024, 9, 28),
        accession_number="0000320193-24-000123",
        primary_document="aapl.htm",
        source_url="https://example.com",
        local_path="2024/aapl.htm",
    )
    with patch("ingest.run.session_scope") as scope, patch(
        "ingest.run.filing_exists", return_value=True
    ):
        scope.return_value.__enter__.return_value = MagicMock()
        assert ingest_filing(filing, downloads_dir=Path("/tmp")) == 0


def test_ingest_filing_processes_new_filing(tmp_path: Path) -> None:
    html_path = tmp_path / "aapl.htm"
    html_path.write_text("<html><body><p>Item 1. Business text.</p></body></html>", encoding="utf-8")
    filing = FilingRecord(
        ticker="AAPL",
        cik="0000320193",
        form_type="10-K",
        filing_date=date(2024, 11, 1),
        report_date=date(2024, 9, 28),
        accession_number="0000320193-24-000123",
        primary_document="aapl.htm",
        source_url="https://example.com",
        local_path="aapl.htm",
    )
    chunks = [TextChunk(0, "chunk", "Item 1. Business", None, 1)]
    with patch("ingest.run.session_scope") as scope, patch(
        "ingest.run.filing_exists", return_value=False
    ), patch("ingest.run.extract_markdown_from_path", return_value="markdown"), patch(
        "ingest.run.chunk_markdown", return_value=chunks
    ), patch("ingest.run.embed_texts", return_value=[[0.1] * 768]), patch(
        "ingest.run.load_filing", return_value="doc-id"
    ) as load:
        scope.return_value.__enter__.return_value = MagicMock()
        count = ingest_filing(filing, downloads_dir=tmp_path)
    assert count == 1
    load.assert_called_once()


def test_run_ingest_counts_success_skip_and_failure(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "filings": [
                    {
                        "ticker": "AAPL",
                        "cik": "0000320193",
                        "form": "10-K",
                        "filing_date": "2024-11-01",
                        "report_date": "2024-09-28",
                        "accession_number": "1",
                        "primary_document": "a.htm",
                        "source_url": "https://example.com/1",
                        "local_path": "a.htm",
                    },
                    {
                        "ticker": "MSFT",
                        "cik": "0000789019",
                        "form": "10-K",
                        "filing_date": "2024-07-30",
                        "report_date": "2024-06-30",
                        "accession_number": "2",
                        "primary_document": "b.htm",
                        "source_url": "https://example.com/2",
                        "local_path": "b.htm",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    filing = load_manifest(manifest_path)[0]

    def ingest_side_effect(filing: FilingRecord, *, downloads_dir: Path) -> int:
        if filing.accession_number == "1":
            return 0
        if filing.accession_number == "2":
            return 3
        raise AssertionError("unexpected filing")

    with patch("ingest.run.ingest_filing", side_effect=ingest_side_effect):
        stats = run_ingest(manifest_path=manifest_path, downloads_dir=tmp_path)

    assert stats.total_filings == 2
    assert stats.skipped_existing == 1
    assert stats.ingested == 1
    assert stats.chunk_count == 3
    assert stats.failures == []


def test_run_ingest_records_failure(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "filings": [
                    {
                        "ticker": "AAPL",
                        "cik": "0000320193",
                        "form": "10-K",
                        "filing_date": "2024-11-01",
                        "report_date": "2024-09-28",
                        "accession_number": "1",
                        "primary_document": "a.htm",
                        "source_url": "https://example.com",
                        "local_path": "a.htm",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with patch("ingest.run.ingest_filing", side_effect=RuntimeError("embed failed")):
        stats = run_ingest(manifest_path=manifest_path, downloads_dir=tmp_path)
    assert stats.failures == ["AAPL 2024 (1): embed failed"]


def test_run_main_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy
    import sys

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"filings": []}), encoding="utf-8")
    run_path = Path(__file__).resolve().parents[2] / "ingest" / "run.py"
    monkeypatch.setattr(sys, "argv", ["run.py", str(manifest)])

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(run_path), run_name="__main__")
    assert exc.value.code == 0


def test_main_success_and_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"filings": []}), encoding="utf-8")
    with patch("ingest.run.run_ingest", return_value=IngestStats(total_filings=0)):
        assert main() == 0

    with patch(
        "ingest.run.run_ingest",
        return_value=IngestStats(total_filings=1, failures=["bad"]),
    ), patch("sys.argv", ["run.py", str(manifest_path)]):
        assert main() == 1
        assert "bad" in capsys.readouterr().out

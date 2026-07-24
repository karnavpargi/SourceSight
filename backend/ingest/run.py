"""End-to-end ingestion orchestrator for local SEC 10-K corpus."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import structlog

from app.config import settings
from ingest.chunk import chunk_markdown
from ingest.db import session_scope
from ingest.embed import embed_texts
from ingest.extract import extract_markdown_from_path
from ingest.load import filing_exists, load_filing

logger = structlog.get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOWNLOADS_DIR = REPO_ROOT / "data" / "downloads"
DEFAULT_MANIFEST_PATH = DEFAULT_DOWNLOADS_DIR / "manifest.json"

COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.",
    "GOOGL": "Alphabet Inc.",
}


@dataclass
class FilingRecord:
    ticker: str
    cik: str
    form_type: str
    filing_date: date
    report_date: date | None
    accession_number: str
    primary_document: str
    source_url: str
    local_path: str

    @classmethod
    def from_manifest_entry(cls, entry: dict[str, str]) -> FilingRecord:
        report_date = _parse_optional_date(entry.get("report_date"))
        return cls(
            ticker=entry["ticker"],
            cik=entry["cik"],
            form_type=entry["form"],
            filing_date=date.fromisoformat(entry["filing_date"]),
            report_date=report_date,
            accession_number=entry["accession_number"],
            primary_document=entry["primary_document"],
            source_url=entry["source_url"],
            local_path=entry["local_path"],
        )

    @property
    def fiscal_year(self) -> int:
        anchor = self.report_date or self.filing_date
        return anchor.year

    @property
    def company_name(self) -> str | None:
        return COMPANY_NAMES.get(self.ticker)


@dataclass
class IngestStats:
    total_filings: int = 0
    skipped_existing: int = 0
    ingested: int = 0
    chunk_count: int = 0
    failures: list[str] = field(default_factory=list)


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> list[FilingRecord]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return [FilingRecord.from_manifest_entry(entry) for entry in manifest["filings"]]


def ingest_filing(
    filing: FilingRecord,
    *,
    downloads_dir: Path = DEFAULT_DOWNLOADS_DIR,
) -> int:
    """Ingest one filing. Returns chunk count, or 0 when skipped."""
    with session_scope() as session:
        if filing_exists(
            session,
            ticker=filing.ticker,
            form_type=filing.form_type,
            fiscal_year=filing.fiscal_year,
            accession_number=filing.accession_number,
        ):
            logger.info(
                "ingest.skip_existing",
                ticker=filing.ticker,
                fiscal_year=filing.fiscal_year,
                accession_number=filing.accession_number,
            )
            return 0

        html_path = downloads_dir / filing.local_path
        markdown = extract_markdown_from_path(html_path)
        chunks = chunk_markdown(markdown)
        if settings.embedding_provider != "none":
            embeddings = embed_texts([chunk.content for chunk in chunks])
        else:
            embeddings = [None] * len(chunks)

        load_filing(
            session,
            ticker=filing.ticker,
            cik=filing.cik,
            company_name=filing.company_name,
            form_type=filing.form_type,
            fiscal_year=filing.fiscal_year,
            accession_number=filing.accession_number,
            filing_date=filing.filing_date,
            report_date=filing.report_date,
            primary_document=filing.primary_document,
            source_url=filing.source_url,
            markdown_content=markdown,
            chunks=chunks,
            embeddings=embeddings,
            extra_metadata={"local_path": filing.local_path},
        )

        logger.info(
            "ingest.complete",
            ticker=filing.ticker,
            fiscal_year=filing.fiscal_year,
            accession_number=filing.accession_number,
            chunk_count=len(chunks),
        )
        return len(chunks)


def run_ingest(
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    downloads_dir: Path = DEFAULT_DOWNLOADS_DIR,
) -> IngestStats:
    filings = load_manifest(manifest_path)
    stats = IngestStats(total_filings=len(filings))

    for filing in filings:
        try:
            chunk_count = ingest_filing(filing, downloads_dir=downloads_dir)
        except Exception as exc:
            message = (
                f"{filing.ticker} {filing.fiscal_year} "
                f"({filing.accession_number}): {exc}"
            )
            stats.failures.append(message)
            logger.exception(
                "ingest.failure",
                ticker=filing.ticker,
                fiscal_year=filing.fiscal_year,
                accession_number=filing.accession_number,
            )
            continue

        if chunk_count == 0:
            stats.skipped_existing += 1
        else:
            stats.ingested += 1
            stats.chunk_count += chunk_count

    logger.info(
        "ingest.summary",
        total_filings=stats.total_filings,
        ingested=stats.ingested,
        skipped_existing=stats.skipped_existing,
        chunk_count=stats.chunk_count,
        failure_count=len(stats.failures),
    )
    return stats


def _parse_optional_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def main() -> int:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )

    manifest_path = DEFAULT_MANIFEST_PATH
    downloads_dir = DEFAULT_DOWNLOADS_DIR
    if len(sys.argv) > 1:
        manifest_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        downloads_dir = Path(sys.argv[2])

    stats = run_ingest(manifest_path=manifest_path, downloads_dir=downloads_dir)
    print(
        f"Ingested {stats.ingested}/{stats.total_filings} filings "
        f"({stats.chunk_count} chunks); "
        f"skipped {stats.skipped_existing} existing; "
        f"{len(stats.failures)} failure(s)."
    )
    for failure in stats.failures:
        print(f"  - {failure}")
    return 1 if stats.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

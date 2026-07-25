from __future__ import annotations

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.source_document import SourceDocument


class CorpusCoverage(BaseModel):
    ticker_years: dict[str, frozenset[int]]

    @property
    def tickers(self) -> frozenset[str]:
        return frozenset(self.ticker_years)

    @property
    def fiscal_years(self) -> frozenset[int]:
        return frozenset(
            year for years in self.ticker_years.values() for year in years
        )

    def prompt_summary(self) -> str:
        return "; ".join(
            f"{ticker}: {','.join(str(year) for year in sorted(years))}"
            for ticker, years in sorted(self.ticker_years.items())
        )


def load_corpus_coverage(session: Session) -> CorpusCoverage:
    statement = select(
        SourceDocument.ticker,
        SourceDocument.fiscal_year,
    ).distinct()
    rows = session.execute(statement).all()
    grouped: dict[str, set[int]] = {}
    for ticker, fiscal_year in rows:
        grouped.setdefault(str(ticker), set()).add(int(fiscal_year))
    return CorpusCoverage(
        ticker_years={ticker: frozenset(years) for ticker, years in grouped.items()}
    )


from dataclasses import dataclass, field
from datetime import date
from uuid import uuid4

from app.retrieval.types import SourcePassage
from app.retrieval.planned import retrieve_for_plan
from app.chat.routing import ValidatedQueryPlan
from app.chat.turn_budget import STANDARD_TURN_BUDGET
from app.chat.usage import TurnUsage
from app.chat.routing import QueryPlan


def _passage(ticker: str, year: int) -> SourcePassage:
    return SourcePassage(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        content=f"{ticker} filing for {year}",
        section="Item 8",
        ticker=ticker,
        company_name=ticker,
        form_type="10-K",
        fiscal_year=year,
        accession_number=f"{ticker}-{year}",
        filing_date=date(year + 1, 1, 31),
        source_url=f"https://example.com/{ticker}/{year}",
        score=1.0,
    )


def _passages(count: int) -> list[SourcePassage]:
    return [_passage("AMZN", 2024) for _ in range(count)]


def _validated_plan(
    *,
    years: list[int] | None = None,
    primary: list[str],
    reserve: list[str] | None = None,
) -> ValidatedQueryPlan:
    return ValidatedQueryPlan(
        plan=QueryPlan(
            route="extractive",
            tickers=["AMZN"],
            fiscal_years=years or [2024],
            topics=["AWS operating income"],
            primary_queries=primary,
            reserve_queries=reserve or [],
            requires_synthesis=False,
        ),
        missing_scope=(),
    )


@dataclass
class RecordingBatchRetriever:
    batches: list[list[SourcePassage]]
    queries: list[list[str]] = field(default_factory=list)

    def search_filings_batch(
        self,
        queries: list[str],
        *,
        limit_per_query: int = 5,
    ) -> list[SourcePassage]:
        self.queries.append(queries)
        return self.batches[len(self.queries) - 1][: limit_per_query * len(queries)]


def test_retrieve_for_plan_uses_reserve_once_when_year_missing() -> None:
    retriever = RecordingBatchRetriever(
        batches=[
            [_passage("AMZN", 2024)],
            [_passage("AMZN", 2023)],
        ]
    )
    result = retrieve_for_plan(
        retriever,
        _validated_plan(
            years=[2023, 2024],
            primary=["AWS 2024"],
            reserve=["AWS 2023"],
        ),
        STANDARD_TURN_BUDGET,
        TurnUsage(),
    )
    assert retriever.queries == [["AWS 2024"], ["AWS 2023"]]
    assert {(p.ticker, p.fiscal_year) for p in result.passages} == {
        ("AMZN", 2023),
        ("AMZN", 2024),
    }


def test_retrieve_for_plan_enforces_unique_passage_cap() -> None:
    result = retrieve_for_plan(
        RecordingBatchRetriever(batches=[_passages(20)]),
        _validated_plan(primary=["q"]),
        STANDARD_TURN_BUDGET,
        TurnUsage(),
    )
    assert len(result.passages) == 8


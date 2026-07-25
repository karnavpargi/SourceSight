from unittest.mock import MagicMock

from sqlalchemy.orm import Session

from app.retrieval.coverage import CorpusCoverage, load_corpus_coverage


def test_coverage_prompt_summary_is_compact_and_sorted() -> None:
    coverage = CorpusCoverage(
        ticker_years={
            "MSFT": frozenset({2023, 2024}),
            "AMZN": frozenset({2022, 2024}),
        }
    )
    assert coverage.prompt_summary() == "AMZN: 2022,2024; MSFT: 2023,2024"


def test_load_corpus_coverage_groups_distinct_rows() -> None:
    session = MagicMock(spec=Session)
    session.execute.return_value.all.return_value = [
        ("AMZN", 2023),
        ("AMZN", 2024),
        ("MSFT", 2024),
    ]
    coverage = load_corpus_coverage(session)
    assert coverage.ticker_years["AMZN"] == frozenset({2023, 2024})
    assert coverage.tickers == frozenset({"AMZN", "MSFT"})

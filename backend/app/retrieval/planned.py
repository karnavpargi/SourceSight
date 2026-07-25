from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.retrieval.types import SourcePassage
from app.chat.turn_budget import TurnBudget
from app.chat.turn_budget import STANDARD_TURN_BUDGET
from app.chat.usage import TurnUsage
from app.chat.routing import ValidatedQueryPlan


@dataclass(frozen=True)
class PlannedRetrieval:
    passages: list[SourcePassage]
    missing_scope: tuple[str, ...]
    expanded: bool


class DocumentRetriever(Protocol):
    def search_filings_batch(self, queries: list[str], *, limit_per_query: int = 5) -> list[SourcePassage]:
        ...


def _dedupe_and_cap(passages: list[SourcePassage], cap: int) -> list[SourcePassage]:
    seen: set = set()
    out: list[SourcePassage] = []
    for p in passages:
        if p.chunk_id in seen:
            continue
        seen.add(p.chunk_id)
        out.append(p)
        if len(out) >= cap:
            break
    return out


def _missing_metadata_scope(plan, passages: list[SourcePassage]) -> list[str]:
    present_tickers = {p.ticker for p in passages}
    present_years = {p.fiscal_year for p in passages}
    missing: list[str] = []
    for ticker in plan.tickers:
        if ticker not in present_tickers:
            missing.append(f"ticker:{ticker}")
    for year in plan.fiscal_years:
        if year not in present_years:
            missing.append(f"year:{year}")
    return missing


def retrieve_for_plan(
    retriever: DocumentRetriever,
    validated: ValidatedQueryPlan,
    budget: TurnBudget,
    usage: TurnUsage,
) -> PlannedRetrieval:
    plan = validated.plan
    primary = plan.primary_queries[: budget.max_searches]
    # Request enough hits per query to allow filling unique-passage cap
    passages = retriever.search_filings_batch(
        primary,
        limit_per_query=max(budget.max_hits_per_search, budget.max_unique_passages),
    )
    for _ in primary:
        usage.record_embedding()
    passages = _dedupe_and_cap(passages, budget.max_unique_passages)

    missing = _missing_metadata_scope(plan, passages)
    reserve_capacity = budget.max_searches - len(primary)
    reserve = plan.reserve_queries[: min(budget.max_reserve_searches, reserve_capacity)]
    expanded = bool(missing and reserve)
    if expanded:
        extra = retriever.search_filings_batch(
            reserve,
            limit_per_query=max(budget.max_hits_per_search, budget.max_unique_passages),
        )
        for _ in reserve:
            usage.record_embedding()
        passages = _dedupe_and_cap([*passages, *extra], budget.max_unique_passages)
        missing = _missing_metadata_scope(plan, passages)

    usage.record_passages(len(passages))
    return PlannedRetrieval(
        passages=passages,
        missing_scope=tuple([*validated.missing_scope, *missing]),
        expanded=expanded,
    )


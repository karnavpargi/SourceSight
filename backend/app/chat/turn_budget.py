from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TurnBudget:
    max_searches: int
    max_reserve_searches: int
    max_hits_per_search: int
    max_unique_passages: int
    router_output_tokens: int = 300
    extractor_output_tokens: int = 2800
    synthesis_output_tokens: int = 2800
    correction_output_tokens: int = 1200
    max_corrections: int = 1


STANDARD_TURN_BUDGET = TurnBudget(
    max_searches=3,
    max_reserve_searches=1,
    max_hits_per_search=5,
    max_unique_passages=8,
)

BROAD_TURN_BUDGET = TurnBudget(
    max_searches=5,
    max_reserve_searches=2,
    max_hits_per_search=5,
    max_unique_passages=15,
    extractor_output_tokens=3500,
)

DEFAULT_TURN_BUDGET = STANDARD_TURN_BUDGET


def budget_for_plan(plan: "QueryPlan") -> TurnBudget:  # type: ignore[name-defined]
    # Broad budget for multi-company plans, standard otherwise.
    return BROAD_TURN_BUDGET if len(set(plan.tickers)) > 1 else STANDARD_TURN_BUDGET

import pytest

from app.chat.routing import QueryPlan, fallback_query_plan
from app.chat.turn_budget import BROAD_TURN_BUDGET, budget_for_plan


def test_query_plan_rejects_too_many_queries() -> None:
    with pytest.raises(Exception):
        QueryPlan(
            route="synthesis",
            tickers=["AMZN"],
            fiscal_years=[2021, 2022],
            topics=["operating income"],
            primary_queries=["q1", "q2", "q3", "q4"],
            reserve_queries=[],
            requires_synthesis=True,
        )


def test_multi_company_plan_uses_broad_budget() -> None:
    plan = QueryPlan(
        route="synthesis",
        tickers=["AMZN", "MSFT"],
        fiscal_years=[2024],
        topics=["capital expenditure"],
        primary_queries=["AMZN capex", "MSFT capex"],
        reserve_queries=["purchase commitments"],
        requires_synthesis=True,
    )
    assert budget_for_plan(plan) == BROAD_TURN_BUDGET


def test_fallback_plan_uses_one_query_and_synthesis() -> None:
    plan = fallback_query_plan("Compare AWS margins")
    assert plan.route == "synthesis"
    assert plan.primary_queries == ["Compare AWS margins"]
    assert plan.requires_synthesis is True

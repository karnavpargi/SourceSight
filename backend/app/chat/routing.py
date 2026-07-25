from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from app.retrieval.coverage import CorpusCoverage

RouteClass = Literal["extractive", "synthesis", "boundary"]


class QueryPlan(BaseModel):
    route: RouteClass
    tickers: list[str] = Field(default_factory=list, max_length=5)
    fiscal_years: list[int] = Field(default_factory=list, max_length=6)
    topics: list[str] = Field(default_factory=list, min_length=1, max_length=8)
    primary_queries: list[str] = Field(default_factory=list, min_length=1, max_length=3)
    reserve_queries: list[str] = Field(default_factory=list, max_length=2)
    requires_synthesis: bool = False

    @model_validator(mode="after")
    def route_matches_synthesis_flag(self) -> "QueryPlan":
        if self.route == "extractive" and self.requires_synthesis:
            raise ValueError("extractive routes cannot require synthesis")
        if self.route == "synthesis" and not self.requires_synthesis:
            raise ValueError("synthesis routes must require synthesis")
        return self


@dataclass(frozen=True)
class ValidatedQueryPlan:
    plan: QueryPlan
    missing_scope: tuple[str, ...]


def fallback_query_plan(question: str) -> QueryPlan:
    return QueryPlan(
        route="synthesis",
        topics=["user question"],
        primary_queries=[question.strip()],
        requires_synthesis=True,
    )


def validate_query_plan(
    plan: QueryPlan,
    coverage: "CorpusCoverage",
) -> ValidatedQueryPlan:
    missing = [
        f"ticker:{ticker}" for ticker in plan.tickers if ticker not in coverage.tickers
    ]
    missing.extend(
        f"year:{year}"
        for year in plan.fiscal_years
        if year not in coverage.fiscal_years
    )
    return ValidatedQueryPlan(plan=plan, missing_scope=tuple(missing))

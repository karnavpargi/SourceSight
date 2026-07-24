from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TurnBudget:
    max_searches: int = 3
    max_hits_per_search: int = 5
    max_unique_passages: int = 8
    max_output_tokens: int = 1500
    max_corrections: int = 1


DEFAULT_TURN_BUDGET = TurnBudget()

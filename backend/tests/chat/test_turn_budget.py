from app.chat.turn_budget import TurnBudget, DEFAULT_TURN_BUDGET


def test_default_turn_budget_matches_spec() -> None:
    budget = DEFAULT_TURN_BUDGET
    assert budget.max_searches == 3
    assert budget.max_hits_per_search == 5
    assert budget.max_unique_passages == 8
    assert budget.max_output_tokens == 2800
    assert budget.max_corrections == 1

import json

import pytest
from pydantic_ai import ModelMessage, ModelResponse, models
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.assistant.router import build_router_prompt, run_query_router
from app.chat.generation import ChatGenerationConfig
from app.chat.routing import QueryPlan
from app.chat.usage import TurnUsage
from app.retrieval.coverage import CorpusCoverage
from tests.evaluation.client_brief_questions import AMAZON_SEGMENTS, CLIENT_BRIEF_CASES

models.ALLOW_MODEL_REQUESTS = False


def _coverage() -> CorpusCoverage:
    return CorpusCoverage(
        ticker_years={
            "AAPL": frozenset({2021, 2022, 2023, 2024, 2025}),
            "AMZN": frozenset({2021, 2022, 2023, 2024, 2025}),
            "GOOGL": frozenset({2021, 2022, 2023, 2024, 2025}),
            "MSFT": frozenset({2021, 2022, 2023, 2024, 2025}),
            "NVDA": frozenset({2021, 2022, 2023, 2024, 2025}),
        }
    )


def test_build_router_prompt_includes_question_verbatim() -> None:
    coverage = _coverage()
    summary = coverage.prompt_summary()
    assert "AMZN: 2021,2022,2023,2024,2025" in summary

    for case in CLIENT_BRIEF_CASES:
        prompt = build_router_prompt(case.question, coverage)
        assert case.question in prompt
        assert summary in prompt


@pytest.mark.anyio
async def test_router_has_no_tools_and_returns_structured_plan() -> None:
    coverage = _coverage()

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.function_tools == []
        rendered = str(messages)
        assert AMAZON_SEGMENTS in rendered
        assert coverage.prompt_summary() in rendered
        return ModelResponse(
            parts=[
                TextPart(
                    json.dumps(
                        {
                            "route": "extractive",
                            "tickers": ["AMZN"],
                            "fiscal_years": [2021, 2022, 2023, 2024, 2025],
                            "topics": ["AWS operating income"],
                            "primary_queries": ["AMZN AWS operating income 2021 2025"],
                            "reserve_queries": ["AMZN segment operating income table"],
                            "requires_synthesis": False,
                        }
                    )
                )
            ]
        )

    plan = await run_query_router(
        AMAZON_SEGMENTS,
        coverage,
        FunctionModel(model_fn),
        ChatGenerationConfig(),
        TurnUsage(),
        max_tokens=300,
    )
    assert isinstance(plan, QueryPlan)
    assert plan.route == "extractive"


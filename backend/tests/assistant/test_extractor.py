import json
from datetime import date
from uuid import UUID

import pytest
from pydantic_ai import ModelMessage, ModelResponse, models
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.assistant.evidence import EvidenceRegistry
from app.assistant.extractor import run_fact_extractor
from app.chat.generation import ChatGenerationConfig
from app.chat.routing import QueryPlan
from app.chat.usage import TurnUsage
from app.retrieval.types import SourcePassage

models.ALLOW_MODEL_REQUESTS = False


def _sample_passage() -> SourcePassage:
    return SourcePassage(
        chunk_id=UUID("11111111-1111-1111-1111-111111111111"),
        document_id=UUID("22222222-2222-2222-2222-222222222222"),
        chunk_index=0,
        content="Sample passage content about AWS operating income.",
        section="Item 1. Business",
        ticker="AMZN",
        company_name="Amazon.com, Inc.",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001234567-24-000001",
        filing_date=date(2024, 2, 15),
        source_url="https://example.com/amzn-10k",
        score=0.9,
    )


@pytest.mark.anyio
async def test_extractor_prompt_includes_aliases_not_chunk_ids_and_extractive_outputs_draft() -> (
    None
):
    evidence = EvidenceRegistry()
    evidence.register([_sample_passage()])
    plan = QueryPlan(
        route="extractive",
        tickers=["AMZN"],
        fiscal_years=[2024],
        topics=["AWS operating income"],
        primary_queries=["AMZN AWS operating income 2024"],
        reserve_queries=[],
        requires_synthesis=False,
    )

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.function_tools == []
        rendered = str(messages)
        assert "E1" in rendered
        assert "11111111-1111-1111-1111-111111111111" not in rendered
        return ModelResponse(
            parts=[
                TextPart(
                    json.dumps(
                        {
                            "facts": [
                                {
                                    "status": "supported",
                                    "ticker": "AMZN",
                                    "fiscal_year": 2024,
                                    "topic": "AWS operating income",
                                    "value": "$1",
                                    "unit": "USD",
                                    "finding": "stub",
                                    "evidence_alias": "E1",
                                }
                            ],
                            "missing_scope": [],
                            "conflicts": [],
                            "draft": {
                                "answer": "AWS operating income was $1 [1].",
                                "citations": [
                                    {
                                        "citation_index": 1,
                                        "evidence_alias": "E1",
                                        "excerpt": "Sample passage content about AWS operating income.",
                                    }
                                ],
                            },
                        }
                    )
                )
            ]
        )

    extraction = await run_fact_extractor(
        "How did AWS operating income change?",
        plan,
        evidence,
        FunctionModel(model_fn),
        ChatGenerationConfig(),
        TurnUsage(),
        max_tokens=800,
    )

    assert extraction.draft is not None


@pytest.mark.anyio
async def test_extractor_synthesis_route_omits_draft() -> None:
    evidence = EvidenceRegistry()
    evidence.register([_sample_passage()])
    plan = QueryPlan(
        route="synthesis",
        tickers=["AMZN"],
        fiscal_years=[2024],
        topics=["AWS operating income"],
        primary_queries=["AMZN AWS operating income 2024"],
        reserve_queries=[],
        requires_synthesis=True,
    )

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.function_tools == []
        rendered = str(messages)
        assert "E1" in rendered
        assert "11111111-1111-1111-1111-111111111111" not in rendered
        return ModelResponse(
            parts=[
                TextPart(
                    json.dumps(
                        {
                            "facts": [
                                {
                                    "status": "supported",
                                    "ticker": "AMZN",
                                    "fiscal_year": 2024,
                                    "topic": "AWS operating income",
                                    "value": "$1",
                                    "unit": "USD",
                                    "finding": "stub",
                                    "evidence_alias": "E1",
                                }
                            ],
                            "missing_scope": [],
                            "conflicts": [],
                        }
                    )
                )
            ]
        )

    extraction = await run_fact_extractor(
        "How did AWS operating income change?",
        plan,
        evidence,
        FunctionModel(model_fn),
        ChatGenerationConfig(),
        TurnUsage(),
        max_tokens=800,
    )

    assert extraction.draft is None

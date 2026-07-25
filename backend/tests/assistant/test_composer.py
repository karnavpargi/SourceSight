import json
from datetime import date
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic_ai import ModelMessage, ModelResponse, models
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.assistant.composer import run_citation_correction, run_direct_fallback, run_synthesis
from app.assistant.evidence import EvidenceRegistry
from app.assistant.facts import FactExtraction, validate_extraction
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
async def test_synthesis_has_no_tools_and_receives_validated_facts_and_compact_evidence_only() -> None:
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
    extraction = FactExtraction(
        facts=[
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
        missing_scope=[],
        conflicts=[],
        draft=None,
    )
    validated = validate_extraction(extraction, evidence, plan.route)

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.function_tools == []
        rendered = str(messages)
        assert "E1" in rendered
        assert "11111111-1111-1111-1111-111111111111" not in rendered
        assert "AMZN: 2021,2022" not in rendered
        assert "ticker_years" not in rendered
        return ModelResponse(
            parts=[
                TextPart(
                    json.dumps(
                        {
                            "answer": "AWS operating income was $1 [1].",
                            "citations": [
                                {
                                    "citation_index": 1,
                                    "evidence_alias": "E1",
                                    "excerpt": "Sample passage content about AWS operating income.",
                                }
                            ],
                        }
                    )
                )
            ]
        )

    draft = await run_synthesis(
        "How did AWS operating income change?",
        plan,
        validated,
        evidence,
        FunctionModel(model_fn),
        ChatGenerationConfig(),
        TurnUsage(),
        max_tokens=1200,
    )

    assert draft.answer
    assert draft.citations


@pytest.mark.anyio
async def test_direct_fallback_has_no_tools_and_uses_evidence_only() -> None:
    evidence = EvidenceRegistry()
    evidence.register([_sample_passage()])

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
                            "answer": "I can’t answer beyond the evidence [1].",
                            "citations": [
                                {
                                    "citation_index": 1,
                                    "evidence_alias": "E1",
                                    "excerpt": "Sample passage content about AWS operating income.",
                                }
                            ],
                        }
                    )
                )
            ]
        )

    draft = await run_direct_fallback(
        "Question?",
        evidence,
        FunctionModel(model_fn),
        ChatGenerationConfig(),
        TurnUsage(),
        max_tokens=500,
    )

    assert draft.answer


@pytest.mark.anyio
async def test_citation_correction_has_no_tools_and_uses_correction_token_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = EvidenceRegistry()
    evidence.register([_sample_passage()])

    captured = SimpleNamespace(max_tokens=None)

    def fake_build_model_settings(config: ChatGenerationConfig, *, max_tokens: int) -> dict[str, object]:
        captured.max_tokens = max_tokens
        return {"temperature": config.temperature, "max_tokens": max_tokens}

    monkeypatch.setattr("app.assistant.composer.build_model_settings", fake_build_model_settings)

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.function_tools == []
        rendered = str(messages)
        assert "grounding_error" in rendered
        assert "failed_answer" in rendered
        assert "E1" in rendered
        return ModelResponse(
            parts=[
                TextPart(
                    json.dumps(
                        {
                            "answer": "Fixed [1].",
                            "citations": [
                                {
                                    "citation_index": 1,
                                    "evidence_alias": "E1",
                                    "excerpt": "Sample passage content about AWS operating income.",
                                }
                            ],
                        }
                    )
                )
            ]
        )

    _ = await run_citation_correction(
        question="Q",
        failed_answer="A",
        grounding_error="error",
        evidence=evidence,
        model=FunctionModel(model_fn),
        model_name="stub-model",
        generation=ChatGenerationConfig(),
        usage=TurnUsage(),
        max_tokens=123,
    )

    assert captured.max_tokens == 123


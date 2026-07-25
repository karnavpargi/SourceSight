from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from uuid import NAMESPACE_DNS, UUID, uuid5

import pytest
from pydantic_ai import ModelMessage, ModelResponse, models
from pydantic_ai.messages import TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from app.chat.generation import ChatGenerationConfig
from app.chat.models_catalog import ResolvedChatModel
from app.chat.orchestrator import REFUSAL_MESSAGE, _run_routed_turn
from app.grounding.validator import grounding_validator
from app.retrieval.coverage import CorpusCoverage
from app.retrieval.types import SourcePassage
from tests.evaluation.client_brief_questions import CLIENT_BRIEF_CASES

models.ALLOW_MODEL_REQUESTS = False

USER_ID = UUID("550e8400-e29b-41d4-a716-446655440000")
THREAD_ID = UUID("770e8400-e29b-41d4-a716-446655440002")

ROUTER_MODEL = ResolvedChatModel(provider="google", model="router-stub")
SYNTHESIS_MODEL = ResolvedChatModel(provider="google", model="synthesis-stub")

_PREVIOUS_CHAT_SENTINEL = "SENTINEL_PREVIOUS_CHAT_SHOULD_NOT_APPEAR"


def _coverage() -> CorpusCoverage:
    years = frozenset({2021, 2022, 2023, 2024, 2025})
    return CorpusCoverage(
        ticker_years={
            "AAPL": years,
            "AMZN": years,
            "GOOGL": years,
            "MSFT": years,
            "NVDA": years,
        }
    )


def _passage(chunk_id: UUID, *, ticker: str) -> SourcePassage:
    return SourcePassage(
        chunk_id=chunk_id,
        document_id=uuid5(NAMESPACE_DNS, f"doc:{ticker}"),
        chunk_index=0,
        content="Stub evidence.",
        section="Item 7. MD&A",
        ticker=ticker,
        company_name=ticker,
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0000000000-00-000000",
        filing_date=date(2025, 2, 1),
        source_url="https://example.com",
        score=1.0,
    )


def _make_passages(*, ticker: str, count: int) -> list[SourcePassage]:
    # Deterministic, many-unique passages so retrieval caps are exercised.
    return [
        _passage(uuid5(NAMESPACE_DNS, f"chunk:{ticker}:{i}"), ticker=ticker)
        for i in range(count)
    ]


def _default_passages() -> list[SourcePassage]:
    return [
        *_make_passages(ticker="AAPL", count=25),
        *_make_passages(ticker="AMZN", count=25),
        *_make_passages(ticker="GOOGL", count=25),
        *_make_passages(ticker="MSFT", count=25),
        *_make_passages(ticker="NVDA", count=25),
    ]


@dataclass
class StubRetriever:
    passages: list[SourcePassage] = field(default_factory=_default_passages)
    queries: list[list[str]] = field(default_factory=list)
    limits: list[int] = field(default_factory=list)
    calls: int = 0

    def search_filings_batch(
        self,
        queries: list[str],
        *,
        limit_per_query: int = 5,
    ) -> list[SourcePassage]:
        assert limit_per_query == 5
        self.limits.append(limit_per_query)
        self.queries.append(list(queries))
        limit = limit_per_query * max(len(queries), 1)
        start = self.calls * limit
        self.calls += 1
        end = start + limit
        return self.passages[start:end]


def _tickers_for_question() -> dict[str, list[str]]:
    # Keep this minimal and deterministic: it drives budget profile (standard vs broad)
    # and ensures the plans validate against the corpus coverage.
    return {
        CLIENT_BRIEF_CASES[0].question: ["AAPL"],
        CLIENT_BRIEF_CASES[1].question: ["AMZN"],
        CLIENT_BRIEF_CASES[2].question: ["NVDA"],
        CLIENT_BRIEF_CASES[3].question: ["MSFT"],
        CLIENT_BRIEF_CASES[4].question: ["GOOGL"],
        CLIENT_BRIEF_CASES[5].question: ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"],
        CLIENT_BRIEF_CASES[6].question: ["AAPL", "NVDA"],
        CLIENT_BRIEF_CASES[7].question: ["AMZN", "GOOGL", "MSFT", "NVDA"],
        CLIENT_BRIEF_CASES[8].question: ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"],
        CLIENT_BRIEF_CASES[9].question: ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"],
    }


def _expected_budget_profile(tickers: list[str]) -> str:
    return "broad" if len(set(tickers)) > 1 else "standard"


def _function_model() -> FunctionModel:
    tickers_for_question = _tickers_for_question()

    def _extract_prompt_json(messages: list[ModelMessage]) -> dict[str, object]:
        for message in messages:
            parts = getattr(message, "parts", None)
            if not isinstance(parts, list):
                continue
            for part in parts:
                content = getattr(part, "content", None)
                if isinstance(content, str) and content.lstrip().startswith("{"):
                    return json.loads(content)
        raise AssertionError("missing JSON prompt part")

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.function_tools == []
        prompt = _extract_prompt_json(messages)
        raw_question = prompt.get("question")
        assert isinstance(raw_question, str) and raw_question
        assert _PREVIOUS_CHAT_SENTINEL not in raw_question
        question = raw_question

        if question not in tickers_for_question:
            raise AssertionError(f"unexpected question: {question!r}")

        expected_route = next(c.expected_route for c in CLIENT_BRIEF_CASES if c.question == question)
        tickers = tickers_for_question[question]

        info_text = repr(info).lower()
        if "query router" in info_text:
            payload = {
                "route": expected_route,
                "tickers": tickers,
                # Force one reserve-expansion path (still bounded) by requesting a year
                # that will not be present in the stub passages.
                "fiscal_years": [2023, 2024],
                "topics": ["topic"],
                # Leave reserve capacity so standard vs broad limits can be asserted.
                "primary_queries": [f"q1:{tickers[0]}", f"q2:{tickers[0]}"],
                "reserve_queries": ["r1", "r2"],
                "requires_synthesis": expected_route == "synthesis",
            }
            return ModelResponse(parts=[TextPart(json.dumps(payload))])

        if "fact extractor" in info_text:
            extraction: dict[str, object] = {
                "facts": [
                    {
                        "status": "supported",
                        "ticker": tickers[0] if tickers else None,
                        "fiscal_year": 2024,
                        "topic": "topic",
                        "value": None,
                        "unit": None,
                        "finding": "Stub finding.",
                        "evidence_alias": "E1",
                    }
                ],
                "missing_scope": [],
                "conflicts": [],
            }
            if expected_route == "extractive":
                extraction["draft"] = {
                    "answer": "Answer [1].",
                    "citations": [
                        {
                            "citation_index": 1,
                            "evidence_alias": "E1",
                            "excerpt": "Stub evidence.",
                        }
                    ],
                }
            return ModelResponse(parts=[TextPart(json.dumps(extraction))])

        if "synthesis writer" in info_text or "direct fallback writer" in info_text:
            if expected_route == "boundary":
                draft = {"answer": REFUSAL_MESSAGE, "citations": []}
            else:
                draft = {
                    "answer": "Synth answer [1].",
                    "citations": [
                        {
                            "citation_index": 1,
                            "evidence_alias": "E1",
                            "excerpt": "Stub evidence.",
                        }
                    ],
                }
            return ModelResponse(parts=[TextPart(json.dumps(draft))])

        raise AssertionError("unexpected stage prompt")

    return FunctionModel(model_fn)


@pytest.mark.anyio
@pytest.mark.parametrize("case", CLIENT_BRIEF_CASES)
async def test_client_brief_offline_routing_regression(case, monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline regression for all ten client-brief questions (no DB, no network)."""
    coverage = _coverage()
    retriever = StubRetriever()
    model = _function_model()
    tickers = _tickers_for_question()[case.question]

    def fake_build_model(_provider: str, _model: str):
        return model

    monkeypatch.setattr("app.chat.orchestrator.resolve_router_model", lambda: ROUTER_MODEL)
    monkeypatch.setattr("app.chat.orchestrator.build_document_agent_model", fake_build_model)

    answer, retrieved_passages, evidence, usage = await _run_routed_turn(
        case.question,
        user_id=USER_ID,
        thread_id=THREAD_ID,
        chat_model=SYNTHESIS_MODEL,
        generation=ChatGenerationConfig(),
        grounding_validator=grounding_validator,
        retriever=retriever,
        coverage=coverage,
        activity=None,
    )

    # Plans validate and budgets are bounded (standard vs broad).
    assert usage.route == case.expected_route
    assert usage.budget_profile == _expected_budget_profile(tickers)

    # Output call counts stay within the design (router + extraction + optional synthesis).
    assert usage.model_calls <= 3
    assert usage.corrections == 0

    # No synthesis for extractive routes; synthesis/boundary at most once.
    if case.expected_route == "extractive":
        assert "synthesis" not in usage.stages
        assert usage.stages["router"].calls == 1
        assert usage.stages["extraction"].calls == 1
    else:
        assert usage.stages["router"].calls == 1
        assert usage.stages["extraction"].calls == 1
        assert usage.stages["synthesis"].calls == 1

    # Every successful fixture finalizes aliases; refusals keep the grounding contract.
    grounding_validator.validate(answer, retrieved_passages)
    if answer.answer == REFUSAL_MESSAGE:
        assert answer.citations == []
        return

    assert answer.citations
    assert "[1]" in answer.answer
    retrieved_ids = {p.chunk_id for p in retrieved_passages}
    assert all(c.chunk_id in retrieved_ids for c in answer.citations)

    # Retrieval stays batched and bounded.
    assert len(retriever.queries) == 2
    assert retriever.limits == [5, 5]
    primary, reserve = retriever.queries
    if usage.budget_profile == "standard":
        assert len(primary) <= 3
        assert len(reserve) <= 1
        cap = 8
    else:
        assert len(primary) <= 5
        assert len(reserve) <= 2
        cap = 15

    # Cap must be enforced on the passages the turn keeps (not a vacuous single-passage stub).
    raw_unique = len({p.chunk_id for p in retrieved_passages})
    assert raw_unique > cap
    kept_unique = len({p.chunk_id for p in evidence.all_passages()})
    assert kept_unique <= cap
    assert usage.passages <= cap
    assert usage.passages == kept_unique


@pytest.mark.anyio
async def test_unknown_evidence_alias_refuses_without_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a draft references an unknown alias, the turn must refuse (not crash)."""
    coverage = _coverage()
    retriever = StubRetriever()

    question = CLIENT_BRIEF_CASES[2].question
    tickers = _tickers_for_question()[question]

    def _extract_prompt_json(messages: list[ModelMessage]) -> dict[str, object]:
        for message in messages:
            parts = getattr(message, "parts", None)
            if not isinstance(parts, list):
                continue
            for part in parts:
                content = getattr(part, "content", None)
                if isinstance(content, str) and content.lstrip().startswith("{"):
                    return json.loads(content)
        raise AssertionError("missing JSON prompt part")

    async def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.function_tools == []
        prompt = _extract_prompt_json(messages)
        raw_question = prompt.get("question")
        assert isinstance(raw_question, str) and raw_question
        assert raw_question == question
        assert _PREVIOUS_CHAT_SENTINEL not in raw_question

        info_text = repr(info).lower()
        if "query router" in info_text:
            payload = {
                "route": "synthesis",
                "tickers": tickers,
                "fiscal_years": [2023, 2024],
                "topics": ["topic"],
                "primary_queries": [f"q1:{tickers[0]}", f"q2:{tickers[0]}"],
                "reserve_queries": ["r1", "r2"],
                "requires_synthesis": True,
            }
            return ModelResponse(parts=[TextPart(json.dumps(payload))])

        if "fact extractor" in info_text:
            extraction: dict[str, object] = {
                "facts": [
                    {
                        "status": "supported",
                        "ticker": tickers[0],
                        "fiscal_year": 2024,
                        "topic": "topic",
                        "value": None,
                        "unit": None,
                        "finding": "Stub finding.",
                        "evidence_alias": "E1",
                    }
                ],
                "missing_scope": [],
                "conflicts": [],
            }
            return ModelResponse(parts=[TextPart(json.dumps(extraction))])

        if "synthesis writer" in info_text:
            draft = {
                "answer": "Synth answer [1].",
                "citations": [
                    {
                        "citation_index": 1,
                        "evidence_alias": "E99",
                        "excerpt": "Invented.",
                    }
                ],
            }
            return ModelResponse(parts=[TextPart(json.dumps(draft))])

        raise AssertionError("unexpected stage prompt")

    model = FunctionModel(model_fn)

    def fake_build_model(_provider: str, _model: str):
        return model

    monkeypatch.setattr("app.chat.orchestrator.resolve_router_model", lambda: ROUTER_MODEL)
    monkeypatch.setattr("app.chat.orchestrator.build_document_agent_model", fake_build_model)

    answer, retrieved_passages, _evidence, usage = await _run_routed_turn(
        question,
        user_id=USER_ID,
        thread_id=THREAD_ID,
        chat_model=SYNTHESIS_MODEL,
        generation=ChatGenerationConfig(),
        grounding_validator=grounding_validator,
        retriever=retriever,
        coverage=coverage,
        activity=None,
    )

    assert usage.route == "synthesis"
    assert usage.corrections == 0
    assert answer.answer == REFUSAL_MESSAGE
    assert answer.citations == []
    grounding_validator.validate(answer, retrieved_passages)


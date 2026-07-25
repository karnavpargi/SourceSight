from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.output import PromptedOutput

from app.assistant.agent import (
    build_document_agent_model,
    infer_usage_model_name,
    token_usage_fields,
)
from app.chat.generation import ChatGenerationConfig, build_model_settings
from app.chat.routing import QueryPlan
from app.chat.usage import TurnUsage
from app.config import settings
from app.retrieval.coverage import CorpusCoverage

__all__ = [
    "build_router_prompt",
    "run_query_router",
]


class _RouterPrompt(BaseModel):
    question: str = Field(min_length=1)
    coverage: str = Field(
        min_length=1,
        description="Compact corpus coverage, e.g. 'AMZN: 2021,2022,2023'.",
    )


ROUTER_INSTRUCTIONS = (
    "You are a query router for an SEC filings assistant.\n"
    "Input is a JSON object with a user question and a compact corpus coverage summary.\n\n"
    "You MUST return a single JSON object matching the QueryPlan schema.\n"
    "- route: one of 'extractive' | 'synthesis' | 'boundary'\n"
    "- tickers: up to 5 tickers mentioned or implied\n"
    "- fiscal_years: up to 6 years in scope\n"
    "- topics: 1–8 short analyst topics\n"
    "- primary_queries: 1–3 focused search queries\n"
    "- reserve_queries: 0–2 backup search queries\n"
    "- requires_synthesis: true only when route is 'synthesis'\n\n"
    "Use 'boundary' when the question asks for proof beyond what filings can support.\n"
    "Do not include any extra keys."
)


router_agent = Agent(
    build_document_agent_model(settings.chat_provider, "catalog-selected"),
    output_type=PromptedOutput(QueryPlan),
    instructions=ROUTER_INSTRUCTIONS,
    retries=0,
)


def build_router_prompt(question: str, coverage: CorpusCoverage) -> str:
    prompt = _RouterPrompt(
        question=question,
        coverage=coverage.prompt_summary(),
    )
    return prompt.model_dump_json()


async def run_query_router(
    question: str,
    coverage: CorpusCoverage,
    model: Model,
    generation: ChatGenerationConfig,
    usage: TurnUsage,
    max_tokens: int,
) -> QueryPlan:
    prompt = build_router_prompt(question, coverage)
    with router_agent.override(model=model):
        run = await router_agent.run(
            prompt,
            model_settings=build_model_settings(generation, max_tokens=max_tokens),
        )
    usage.add_model_usage(
        stage="router",
        model=infer_usage_model_name(model),
        **token_usage_fields(run),
    )
    return run.output


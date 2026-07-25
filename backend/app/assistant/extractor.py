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
from app.assistant.evidence import CompactEvidence, EvidenceRegistry
from app.assistant.facts import FactExtraction
from app.chat.generation import ChatGenerationConfig, build_model_settings
from app.chat.routing import QueryPlan
from app.chat.usage import TurnUsage
from app.config import settings

__all__ = ["run_fact_extractor"]


class _ExtractorPrompt(BaseModel):
    question: str = Field(min_length=1)
    plan: QueryPlan
    evidence: list[CompactEvidence] = Field(
        default_factory=list,
        description="Compact evidence rows keyed by alias; no chunk UUIDs.",
    )


EXTRACTOR_INSTRUCTIONS = (
    "You are a fact extractor for an SEC filings assistant.\n"
    "Input is a JSON object with the user question, a QueryPlan, and compact evidence rows.\n\n"
    "You MUST return a single JSON object matching the FactExtraction schema.\n"
    "- ExtractedFact.status: 'supported' | 'missing' | 'conflicting'\n"
    "- supported/conflicting facts MUST include evidence_alias referencing one of the aliases.\n"
    "- missing facts MUST NOT cite evidence.\n\n"
    "Draft rules:\n"
    "- If plan.route == 'extractive': include a GroundedDraft under 'draft'.\n"
    "- If plan.route in {'synthesis','boundary'}: omit 'draft' (set it to null or leave it out).\n\n"
    "Never include chunk UUIDs. Cite evidence only by alias (E1, E2, ...)."
)


extractor_agent = Agent(
    build_document_agent_model(settings.chat_provider, "catalog-selected"),
    output_type=PromptedOutput(FactExtraction),
    instructions=EXTRACTOR_INSTRUCTIONS,
    retries=0,
)


async def run_fact_extractor(
    question: str,
    plan: QueryPlan,
    evidence: EvidenceRegistry,
    model: Model,
    generation: ChatGenerationConfig,
    usage: TurnUsage,
    max_tokens: int,
) -> FactExtraction:
    prompt = _ExtractorPrompt(
        question=question,
        plan=plan,
        evidence=evidence.compact_dump(),
    ).model_dump_json()
    with extractor_agent.override(model=model):
        run = await extractor_agent.run(
            prompt,
            model_settings=build_model_settings(generation, max_tokens=max_tokens),
        )
    usage.add_model_usage(
        stage="extraction",
        model=infer_usage_model_name(model),
        **token_usage_fields(run),
    )
    return run.output


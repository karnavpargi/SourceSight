from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.output import PromptedOutput

from app.assistant.agent import build_document_agent_model, infer_usage_model_name, token_usage_fields
from app.assistant.evidence import CompactEvidence, EvidenceRegistry
from app.assistant.facts import ExtractedFact, ValidatedExtraction
from app.assistant.outputs import GroundedDraft
from app.chat.generation import ChatGenerationConfig, build_model_settings
from app.chat.routing import QueryPlan
from app.chat.usage import TurnUsage
from app.config import settings

__all__ = [
    "run_synthesis",
    "run_direct_fallback",
    "run_citation_correction",
]


class _ValidatedExtractionPrompt(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)
    missing_scope: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)


class _SynthesisPrompt(BaseModel):
    question: str = Field(min_length=1)
    plan: QueryPlan
    extraction: _ValidatedExtractionPrompt
    evidence: list[CompactEvidence] = Field(default_factory=list)


class _FallbackPrompt(BaseModel):
    question: str = Field(min_length=1)
    evidence: list[CompactEvidence] = Field(default_factory=list)


class _CorrectionPrompt(BaseModel):
    question: str = Field(min_length=1)
    failed_answer: str = Field(min_length=1)
    grounding_error: str = Field(min_length=1)
    evidence: list[CompactEvidence] = Field(default_factory=list)


SYNTHESIS_INSTRUCTIONS = (
    "You are a synthesis writer for an SEC filings assistant.\n"
    "Input is JSON with the user question, a QueryPlan, validated extracted facts, and compact evidence rows.\n\n"
    "You MUST return a single JSON object matching the GroundedDraft schema:\n"
    "- answer: prose with inline [n] markers\n"
    "- citations: list of {citation_index, evidence_alias, excerpt}\n\n"
    "Rules:\n"
    "- Only cite evidence_alias values that exist in the evidence list.\n"
    "- Do not invent facts not supported by evidence.\n"
    "- Use the validated facts as your starting point; if a fact is 'missing', say so explicitly.\n"
    "- Never include chunk UUIDs."
)


FALLBACK_INSTRUCTIONS = (
    "You are a direct fallback writer for an SEC filings assistant.\n"
    "Input is JSON with the user question and compact evidence rows.\n\n"
    "Return a GroundedDraft. If evidence is insufficient, refuse to infer beyond the filings.\n"
    "Cite only by evidence alias and include short excerpts. Never include chunk UUIDs."
)


CORRECTION_INSTRUCTIONS = (
    "You are a citation correction pass for an SEC filings assistant.\n"
    "Input is JSON containing the question, the failed answer text, the grounding validation error, "
    "and a fixed set of compact evidence rows.\n\n"
    "Return a corrected GroundedDraft that satisfies the grounding error. Do not retrieve new evidence. "
    "Only adjust wording and citations using the provided evidence aliases."
)


synthesis_agent = Agent(
    build_document_agent_model(settings.chat_provider, "catalog-selected"),
    output_type=PromptedOutput(GroundedDraft),
    instructions=SYNTHESIS_INSTRUCTIONS,
    retries=0,
)

fallback_agent = Agent(
    build_document_agent_model(settings.chat_provider, "catalog-selected"),
    output_type=PromptedOutput(GroundedDraft),
    instructions=FALLBACK_INSTRUCTIONS,
    retries=0,
)

correction_agent = Agent(
    build_document_agent_model(settings.chat_provider, "catalog-selected"),
    output_type=PromptedOutput(GroundedDraft),
    instructions=CORRECTION_INSTRUCTIONS,
    retries=0,
)


async def run_synthesis(
    question: str,
    plan: QueryPlan,
    validated_extraction: ValidatedExtraction,
    evidence: EvidenceRegistry,
    model: Model,
    generation: ChatGenerationConfig,
    usage: TurnUsage,
    max_tokens: int,
) -> GroundedDraft:
    prompt = _SynthesisPrompt(
        question=question,
        plan=plan,
        extraction=_ValidatedExtractionPrompt(
            facts=validated_extraction.facts,
            missing_scope=list(validated_extraction.missing_scope),
            conflicts=list(validated_extraction.conflicts),
            validation_errors=list(validated_extraction.validation_errors),
        ),
        evidence=evidence.compact_dump(),
    ).model_dump_json()
    with synthesis_agent.override(model=model):
        run = await synthesis_agent.run(
            prompt,
            model_settings=build_model_settings(generation, max_tokens=max_tokens),
        )
    usage.add_model_usage(
        stage="synthesis",
        model=infer_usage_model_name(model),
        **token_usage_fields(run),
    )
    return run.output


async def run_direct_fallback(
    question: str,
    evidence: EvidenceRegistry,
    model: Model,
    generation: ChatGenerationConfig,
    usage: TurnUsage,
    max_tokens: int,
) -> GroundedDraft:
    prompt = _FallbackPrompt(
        question=question,
        evidence=evidence.compact_dump(),
    ).model_dump_json()
    with fallback_agent.override(model=model):
        run = await fallback_agent.run(
            prompt,
            model_settings=build_model_settings(generation, max_tokens=max_tokens),
        )
    usage.add_model_usage(
        stage="fallback",
        model=infer_usage_model_name(model),
        **token_usage_fields(run),
    )
    return run.output


async def run_citation_correction(
    question: str,
    failed_answer: str,
    grounding_error: object,
    evidence: EvidenceRegistry,
    model: Model,
    model_name: str,
    generation: ChatGenerationConfig,
    usage: TurnUsage,
    max_tokens: int,
) -> GroundedDraft:
    prompt = _CorrectionPrompt(
        question=question,
        failed_answer=failed_answer,
        grounding_error=str(grounding_error),
        evidence=evidence.compact_dump(),
    ).model_dump_json()
    with correction_agent.override(model=model):
        run = await correction_agent.run(
            prompt,
            model_settings=build_model_settings(generation, max_tokens=max_tokens),
        )
    usage.add_model_usage(
        stage="correction",
        model=model_name,
        **token_usage_fields(run),
    )
    return run.output


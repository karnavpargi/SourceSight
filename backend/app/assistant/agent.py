"""PydanticAI document assistant agent."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.assistant.deps import DocumentAgentDeps
from app.assistant.outputs import GroundedAnswer
from app.config import settings
from app.retrieval.types import RetrievalResult, SourcePassage

INSTRUCTIONS_PATH = Path(__file__).with_name("instructions.md")

__all__ = ["build_document_agent_model", "document_agent", "load_instructions", "openai_model_name"]


def load_instructions() -> str:
    return INSTRUCTIONS_PATH.read_text(encoding="utf-8")


def openai_model_name() -> str:
    return f"openai:{settings.openai_chat_model}"


def build_document_agent_model() -> OpenAIChatModel:
    return OpenAIChatModel(
        settings.openai_chat_model,
        provider=OpenAIProvider(api_key=settings.openai_api_key),
    )


document_agent = Agent(
    build_document_agent_model(),
    deps_type=DocumentAgentDeps,
    output_type=GroundedAnswer,
    instructions=load_instructions(),
)


@document_agent.tool
def search_filings(
    ctx: RunContext[DocumentAgentDeps],
    query: str,
    limit: int = 10,
) -> RetrievalResult:
    """Search indexed SEC filing chunks for passages relevant to the query."""
    return ctx.deps.retriever.search_filings(query, limit=limit)


@document_agent.tool
def read_chunk(ctx: RunContext[DocumentAgentDeps], chunk_id: UUID) -> SourcePassage:
    """Load one filing chunk by ID with source document metadata."""
    return ctx.deps.retriever.read_chunk(chunk_id)


@document_agent.tool
def read_surrounding_chunks(
    ctx: RunContext[DocumentAgentDeps],
    chunk_id: UUID,
    window: int = 1,
) -> list[SourcePassage]:
    """Load neighboring chunks around a target chunk for additional context."""
    return ctx.deps.retriever.read_surrounding_chunks(chunk_id, window=window)

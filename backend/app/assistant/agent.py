"""PydanticAI document assistant agent."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import PromptedOutput
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from app.assistant.deps import DocumentAgentDeps
from app.assistant.outputs import GroundedAnswer
from app.config import ChatProvider, settings
from app.retrieval.types import RetrievalResult, SourcePassage

INSTRUCTIONS_PATH = Path(__file__).with_name("instructions.md")

# Placeholder model name — every chat turn overrides the model via orchestrator.
_AGENT_PLACEHOLDER_MODEL = "catalog-selected"

__all__ = [
    "build_document_agent_model",
    "chat_model_name",
    "document_agent",
    "load_instructions",
]


def _optional_int(value: int | None, default: int) -> int:
    # Local models often emit explicit null for defaulted tool args.
    return default if value is None else value


def load_instructions() -> str:
    return INSTRUCTIONS_PATH.read_text(encoding="utf-8")


def chat_model_name(provider: ChatProvider, model: str) -> str:
    return f"{provider}:{model}"


def build_document_agent_model(provider: ChatProvider, model: str) -> Model:
    if not model.strip():
        raise ValueError("model is required")

    if provider == "google":
        return GoogleModel(
            model,
            provider=GoogleProvider(api_key=settings.google_api_key),
        )

    if provider == "local":
        return OpenAIChatModel(
            model,
            provider=OllamaProvider(base_url=settings.ollama_openai_base_url()),
        )

    return OpenAIChatModel(
        model,
        provider=OpenAIProvider(
            base_url=settings.opencode_base_url,
            api_key=settings.opencode_api_key,
        ),
    )


document_agent = Agent(
    build_document_agent_model(settings.chat_provider, _AGENT_PLACEHOLDER_MODEL),
    deps_type=DocumentAgentDeps,
    output_type=PromptedOutput(GroundedAnswer),
    instructions=load_instructions(),
    retries=3,
)


@document_agent.tool
def search_filings(
    ctx: RunContext[DocumentAgentDeps],
    query: str,
    limit: int | None = 10,
) -> RetrievalResult:
    """Search indexed SEC filing chunks for passages relevant to the query."""
    return ctx.deps.retriever.search_filings(query, limit=_optional_int(limit, 10))


@document_agent.tool
def read_chunk(ctx: RunContext[DocumentAgentDeps], chunk_id: UUID) -> SourcePassage:
    """Load one filing chunk by ID with source document metadata."""
    return ctx.deps.retriever.read_chunk(chunk_id)


@document_agent.tool
def read_surrounding_chunks(
    ctx: RunContext[DocumentAgentDeps],
    chunk_id: UUID,
    window: int | None = 1,
) -> list[SourcePassage]:
    """Load neighboring chunks around a target chunk for additional context."""
    return ctx.deps.retriever.read_surrounding_chunks(
        chunk_id,
        window=_optional_int(window, 1),
    )

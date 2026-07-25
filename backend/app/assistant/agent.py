"""PydanticAI document assistant agent."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.output import PromptedOutput
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from app.assistant.deps import DocumentAgentDeps
from app.assistant.evidence import CompactEvidence
from app.assistant.outputs import GroundedDraft
from app.config import ChatProvider, settings
from app.http_client import get_async_client

INSTRUCTIONS_PATH = Path(__file__).with_name("instructions.md")

# Placeholder model name — every chat turn overrides the model via orchestrator.
_AGENT_PLACEHOLDER_MODEL = "catalog-selected"

__all__ = [
    "build_document_agent_model",
    "chat_model_name",
    "document_agent",
    "infer_usage_model_name",
    "load_instructions",
    "token_usage_fields",
]

def load_instructions() -> str:
    return INSTRUCTIONS_PATH.read_text(encoding="utf-8")


def chat_model_name(provider: ChatProvider, model: str) -> str:
    return f"{provider}:{model}"


def build_document_agent_model(provider: ChatProvider, model: str) -> Model:
    if not model.strip():
        raise ValueError("model is required")

    http_client = get_async_client()

    if provider == "google":
        return GoogleModel(
            model,
            provider=GoogleProvider(
                api_key=settings.google_api_key,
                http_client=http_client,
            ),
        )

    if provider == "local":
        return OpenAIChatModel(
            model,
            provider=OllamaProvider(
                base_url=settings.ollama_openai_base_url(),
                http_client=http_client,
            ),
        )

    return OpenAIChatModel(
        model,
        provider=OpenAIProvider(
            base_url=settings.opencode_base_url,
            api_key=settings.opencode_api_key,
            http_client=http_client,
        ),
    )


document_agent = Agent(
    build_document_agent_model(settings.chat_provider, _AGENT_PLACEHOLDER_MODEL),
    deps_type=DocumentAgentDeps,
    output_type=PromptedOutput(GroundedDraft),
    instructions=load_instructions(),
    retries=1,
)


def _search_filings_impl(
    deps: DocumentAgentDeps,
    queries: list[str],
) -> list[CompactEvidence]:
    """Core implementation for the search_filings tool."""
    budget = deps.budget

    # Correction runs must not perform additional retrieval, and a zero-search
    # budget disables searching entirely.
    if budget.max_searches <= 0 or getattr(deps, "correction_mode", False):
        return []

    # Clean and cap queries against the remaining per-turn budget.
    remaining = max(budget.max_searches - deps.search_count, 0)
    if remaining <= 0:
        return []

    cleaned = [q.strip() for q in queries if q and q.strip()][:remaining]
    if not cleaned:
        return []

    # Track total cleaned queries issued this turn.
    deps.search_count += len(cleaned)

    passages = deps.retriever.search_filings_batch(
        cleaned,
        limit_per_query=budget.max_hits_per_search,
    )
    if settings.embedding_provider != "none":
        # Approximate embedding accounting: one embedding call per cleaned query.
        for _ in cleaned:
            deps.usage.record_embedding()
    compact = deps.evidence.register(passages)
    deps.usage.record_passages(len(deps.evidence.all_passages()))
    return compact


@document_agent.tool
def search_filings(
    ctx: RunContext[DocumentAgentDeps],
    queries: list[str],
) -> list[CompactEvidence]:
    """Search filings with 1–3 focused queries; returns compact evidence aliases."""
    return _search_filings_impl(ctx.deps, queries)


def infer_usage_model_name(model: object) -> str:
    """Derive a stable model identifier for usage attribution."""
    value = getattr(model, "model_name", None)
    if isinstance(value, str) and value.strip():
        return value
    value = getattr(model, "model", None)
    if isinstance(value, str) and value.strip():
        return value
    return "function"


def token_usage_fields(run: object) -> dict[str, int | None]:
    """Extract input/output token counts from a pydantic-ai run object."""
    usage = getattr(run, "usage", None)
    if callable(usage):
        usage = usage()
    if usage is None:
        return {"input_tokens": None, "output_tokens": None}

    input_tokens = getattr(usage, "input_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "request_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "response_tokens", None)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens}



"""PydanticAI document assistant."""

from app.assistant.agent import build_document_agent_model, document_agent, load_instructions
from app.assistant.deps import DocumentAgentDeps, DocumentRetriever, GroundingValidator
from app.assistant.outputs import Citation, GroundedAnswer, SourcePassage

__all__ = [
    "Citation",
    "DocumentAgentDeps",
    "DocumentRetriever",
    "GroundedAnswer",
    "GroundingValidator",
    "SourcePassage",
    "build_document_agent_model",
    "document_agent",
    "load_instructions",
]

"""AI SDK wire-format helpers and internal chat message models."""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.assistant.outputs import Citation, GroundedAnswer
from app.retrieval.types import SourcePassage

ChatRole = Literal["user", "assistant", "system"]


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class CitationPart(BaseModel):
    type: Literal["citation"] = "citation"
    citation_index: int = Field(ge=1)
    chunk_id: UUID
    excerpt: str


class SourcePassagePart(BaseModel):
    type: Literal["source-passage"] = "source-passage"
    passage: SourcePassage


ChatUIPart = Annotated[
    TextPart | CitationPart | SourcePassagePart,
    Field(discriminator="type"),
]


class ChatUIMessage(BaseModel):
    id: str
    role: ChatRole
    parts: list[ChatUIPart] = Field(default_factory=list)


def parse_ui_message(data: dict) -> ChatUIMessage:
    """Parse one AI SDK UI message from request or persistence wire JSON."""
    message_id = data.get("id")
    if not isinstance(message_id, str) or not message_id:
        raise ValueError("Message id is required.")

    role = data.get("role")
    if role not in ("user", "assistant", "system"):
        raise ValueError(f"Unsupported message role: {role!r}")

    parts = _parse_parts(data)
    return ChatUIMessage(id=message_id, role=role, parts=parts)


def parse_ui_messages(data: list[dict]) -> list[ChatUIMessage]:
    return [parse_ui_message(message) for message in data]


def ui_message_to_wire(message: ChatUIMessage) -> dict:
    """Serialize an internal UI message back to AI SDK wire JSON."""
    return message.model_dump(mode="json")


def message_text(message: ChatUIMessage) -> str:
    """Concatenate text parts from a UI message."""
    return "".join(part.text for part in message.parts if isinstance(part, TextPart))


def grounded_answer_to_ui_message(answer: GroundedAnswer, *, message_id: str) -> ChatUIMessage:
    """Build a persisted/streamed assistant UI message from a grounded answer."""
    parts: list[ChatUIPart] = [TextPart(text=answer.answer)]
    parts.extend(_citation_to_part(citation) for citation in answer.citations)
    parts.extend(SourcePassagePart(passage=passage) for passage in answer.cited_passages)
    return ChatUIMessage(id=message_id, role="assistant", parts=parts)


def extract_latest_user_text(messages: list[dict]) -> str:
    """Return the text of the most recent user message from AI SDK UI messages."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue

        try:
            text = message_text(parse_ui_message(message))
        except ValueError:
            text = _legacy_content_text(message)

        if text:
            return text

    raise ValueError("No user message found in request.")


def _parse_parts(data: dict) -> list[ChatUIPart]:
    raw_parts = data.get("parts")
    if isinstance(raw_parts, list) and raw_parts:
        return [_parse_part(part) for part in raw_parts if isinstance(part, dict)]

    content = data.get("content")
    if isinstance(content, str) and content.strip():
        return [TextPart(text=content.strip())]

    return []


def _parse_part(data: dict) -> ChatUIPart:
    part_type = data.get("type")
    if part_type == "text":
        return TextPart.model_validate(data)
    if part_type == "citation":
        return CitationPart.model_validate(data)
    if part_type == "source-passage":
        return SourcePassagePart.model_validate(data)
    raise ValueError(f"Unsupported message part type: {part_type!r}")


def _citation_to_part(citation: Citation) -> CitationPart:
    return CitationPart(
        citation_index=citation.citation_index,
        chunk_id=citation.chunk_id,
        excerpt=citation.excerpt,
    )


def _legacy_content_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    return ""

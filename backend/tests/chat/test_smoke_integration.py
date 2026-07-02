"""End-to-end smoke test for grounded chat answers."""

from __future__ import annotations

from uuid import UUID

import pytest

from app.assistant.agent import document_agent
from app.assistant.deps import DocumentAgentDeps
from app.database.session import session_scope
from app.grounding.validator import grounding_validator
from app.retrieval.document_retriever import SessionDocumentRetriever

pytestmark = pytest.mark.integration

USER_ID = UUID("550e8400-e29b-41d4-a716-446655440000")
THREAD_ID = UUID("770e8400-e29b-41d4-a716-446655440002")

AWS_QUESTION = (
    "For Amazon, compare AWS operating income against North America and International "
    "from 2021-2025. Which years did AWS appear to fund weaker profitability elsewhere?"
)


@pytest.mark.anyio
async def test_client_brief_aws_question_returns_cited_answer(
    ingested_corpus: None,
    ollama_embeddings: None,
) -> None:
    with session_scope() as session:
        retriever = SessionDocumentRetriever(session)
        deps = DocumentAgentDeps(
            user_id=USER_ID,
            thread_id=THREAD_ID,
            retriever=retriever,
            grounding_validator=grounding_validator,
        )
        run = await document_agent.run(AWS_QUESTION, deps=deps)

    answer = run.output

    assert answer.answer.strip()
    assert len(answer.citations) >= 1
    assert len(answer.cited_passages) >= 1

    cited_chunk_ids = {passage.chunk_id for passage in answer.cited_passages}
    for citation in answer.citations:
        assert citation.chunk_id in cited_chunk_ids

    assert any(passage.ticker == "AMZN" for passage in answer.cited_passages)

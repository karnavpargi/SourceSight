from __future__ import annotations

from datetime import date
from unittest.mock import Mock
from uuid import UUID

from app.chat.orchestrator import _RecordingRetriever
from app.chat.turn_activity import TurnActivityEmitter
from app.retrieval.types import SourcePassage

CHUNK_ID = UUID("11111111-1111-1111-1111-111111111111")
DOCUMENT_ID = UUID("22222222-2222-2222-2222-222222222222")


def _passage() -> SourcePassage:
    return SourcePassage(
        chunk_id=CHUNK_ID,
        document_id=DOCUMENT_ID,
        chunk_index=3,
        content="Data Center demand increased.",
        section="Item 1. Business",
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001045810-24-000029",
        filing_date=date(2024, 2, 21),
        source_url="https://example.com/nvda-10k",
        score=0.42,
    )


def test_search_filings_batch_records_passages_and_emits_activity_updates() -> None:
    passage = _passage()
    inner = Mock()
    inner.search_filings_batch.return_value = [passage, passage]
    emitter = TurnActivityEmitter()
    recorder = _RecordingRetriever(inner=inner, activity=emitter)

    step_id = emitter.start("retrieve", "Searching indexed filings...")
    emitter.bind_active_tool(step_id)
    result = recorder.search_filings_batch(["NVDA Data Center demand"], limit_per_query=5)

    assert result == [passage, passage]
    assert recorder.retrieved_passages == [passage]
    updates = emitter.drain()
    assert any(update.phase == "update" for update in updates)

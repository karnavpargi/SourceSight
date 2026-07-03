"""Corpus status routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.auth.dependencies import get_current_user
from app.auth.types import CurrentUser
from app.database.corpus import corpus_counts
from app.database.session import session_scope

router = APIRouter(tags=["corpus"])


class CorpusStatusResponse(BaseModel):
    document_count: int
    chunk_count: int
    ready: bool


@router.get("/corpus/status", response_model=CorpusStatusResponse)
async def get_corpus_status(
    user: CurrentUser = Depends(get_current_user),
) -> CorpusStatusResponse:
    del user
    with session_scope() as session:
        document_count, chunk_count = corpus_counts(session)

    return CorpusStatusResponse(
        document_count=document_count,
        chunk_count=chunk_count,
        ready=document_count > 0 and chunk_count > 0,
    )

"""Re-embed existing document chunks in place (e.g. after switching embedding providers)."""

from __future__ import annotations

import argparse
import sys
import uuid

import structlog
from sqlalchemy import select, update

from app.database.document_chunk import DocumentChunk
from ingest.db import session_scope
from ingest.embed import embed_texts

logger = structlog.get_logger(__name__)

DEFAULT_BATCH_SIZE = 32


def reembed_chunks(
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    document_id: uuid.UUID | None = None,
) -> tuple[int, int]:
    """Update embeddings for all chunks (or one document). Returns (updated, total)."""
    updated = 0
    total = 0

    with session_scope() as session:
        stmt = (
            select(DocumentChunk.id, DocumentChunk.content)
            .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
        )
        if document_id is not None:
            stmt = stmt.where(DocumentChunk.document_id == document_id)

        rows = session.execute(stmt).all()
        total = len(rows)
        if total == 0:
            return 0, 0

        for start in range(0, total, batch_size):
            batch = rows[start : start + batch_size]
            texts = [content for _, content in batch]
            vectors = embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")

            for (chunk_id, _), vector in zip(batch, vectors, strict=True):
                session.execute(
                    update(DocumentChunk)
                    .where(DocumentChunk.id == chunk_id)
                    .values(embedding=vector)
                )
                updated += 1

            session.commit()
            logger.info(
                "reembed.batch_complete",
                updated=updated,
                total=total,
                batch_size=len(batch),
            )

    return updated, total


def main(argv: list[str] | None = None) -> int:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )

    parser = argparse.ArgumentParser(
        description="Re-embed existing document_chunks with the configured embedding provider.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Chunks per embedding API call (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--document-id",
        type=uuid.UUID,
        default=None,
        help="Re-embed only chunks for this source_documents.id",
    )
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        print("batch-size must be at least 1", file=sys.stderr)
        return 2

    updated, total = reembed_chunks(
        batch_size=args.batch_size,
        document_id=args.document_id,
    )
    print(f"Re-embedded {updated}/{total} chunks.")
    return 0 if updated == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

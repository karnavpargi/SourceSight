# Google AI Studio as primary AI provider

**Date:** 2026-07-24
**Status:** Approved

## Goal

Make Google AI Studio the default for chat and embeddings while keeping OpenCode and local Ollama available when configured.

## Current state

- `backend/app/config.py` already defaults `chat_provider` to `"google"` and requires `GOOGLE_API_KEY` when that provider is selected.
- PydanticAI already wires Google via `GoogleModel` + `GoogleProvider(api_key=settings.google_api_key)`.
- `/chat/providers` returns `default_provider` from `settings.chat_provider`.
- Google embeddings already use `gemini-embedding-001`, request 768-dimensional vectors, and distinguish retrieval documents from retrieval queries.
- Drift: `backend/.env.example` and `backend/README.md` still document OpenCode for chat and Ollama for embeddings.
- Local `backend/.env` already has `CHAT_PROVIDER=google` and a Google API key set (not committed).
- Existing stored chunks were embedded with Ollama and must be re-embedded before Google query embeddings are used.

## Scope

**In**

- Document Google AI Studio as the primary/default chat provider in `.env.example` and README.
- Set Google as the default embedding provider while retaining the existing 768-dimensional database schema.
- Re-embed all existing document chunks with `gemini-embedding-001`.
- Keep other providers selectable when their keys/flags are present.

**Out**

- No agent/runtime code changes.
- No removal of OpenCode or Ollama.
- No embedding dimension or database schema migration.
- Do not commit secrets from `.env`.

## Design

1. Set `CHAT_PROVIDER=google` and `EMBEDDING_PROVIDER=google` in local `backend/.env`.
2. Set the same provider defaults in `backend/.env.example`.
3. Keep `GOOGLE_EMBEDDING_MODEL=gemini-embedding-001` and `EMBEDDING_DIMENSIONS=768`.
4. Update `backend/README.md` so Google AI Studio is the documented primary chat and embedding provider; OpenCode and Ollama remain optional alternatives.
5. Run `uv run python -m ingest.reembed` from `backend/`. It reads all chunks, requests `RETRIEVAL_DOCUMENT` embeddings in batches, and commits each completed batch.
6. Verify that every stored chunk was updated and run the relevant non-integration tests.

If re-embedding fails partway through, already committed batches remain valid Google vectors. Re-run the same command to replace all chunks consistently; do not serve retrieval traffic while vectors may be mixed.

## Success criteria

- Fresh setups copying `.env.example` get Google as the documented default.
- README production stack matches that default.
- With `GOOGLE_API_KEY` set and `CHAT_PROVIDER=google`, `GET /chat/providers` returns `default_provider: "google"`.
- Query-time embeddings and all stored document embeddings use `gemini-embedding-001` with 768 dimensions.
- The re-embed command reports `Re-embedded N/N chunks.` and exits successfully.
- OpenCode/local still appear when configured.

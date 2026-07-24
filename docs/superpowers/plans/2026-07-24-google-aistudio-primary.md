# Google AI Studio Primary Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Google AI Studio the default chat and embedding provider, then re-embed all stored chunks with `gemini-embedding-001` at 768 dimensions.

**Architecture:** No new runtime modules. Switch operator and documented defaults to `CHAT_PROVIDER=google` and `EMBEDDING_PROVIDER=google`, keep OpenCode/Ollama optional, and use the existing `ingest.reembed` batch job so query embeddings and stored vectors share the same Google embedding space.

**Tech Stack:** FastAPI settings (`pydantic-settings`), Google Generative Language API via existing `ingest.providers.google`, SQLAlchemy + Supabase Postgres `pgvector(768)`, `uv`/`pytest`.

## Global Constraints

- Keep OpenCode and local Ollama available when configured.
- Keep `GOOGLE_EMBEDDING_MODEL=gemini-embedding-001` and `EMBEDDING_DIMENSIONS=768`.
- No embedding-dimension or database schema migration.
- Do not commit secrets from `backend/.env`.
- Do not serve retrieval traffic while vectors may be mixed; if re-embed fails mid-run, re-run until `N/N` succeeds.

## File map

| File | Responsibility |
|------|----------------|
| `backend/.env` | Local operator secrets and live provider selection (gitignored) |
| `backend/.env.example` | Documented defaults for new setups |
| `backend/app/config.py` | Python default for `embedding_provider` when env is unset |
| `backend/README.md` | Production stack and rollout docs |
| `backend/ingest/reembed.py` | Existing re-embed job (run only; no code changes expected) |

---

### Task 1: Point local env and settings defaults at Google

**Files:**
- Modify: `backend/.env`
- Modify: `backend/.env.example`
- Modify: `backend/app/config.py`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Consumes: existing `Settings.chat_provider`, `Settings.embedding_provider`, `Settings.google_api_key`
- Produces: live and documented defaults of `chat_provider="google"` and `embedding_provider="google"` with unchanged `embedding_dimensions=768`

- [ ] **Step 1: Confirm local `.env` keys without printing secrets**

Run from repo root:

```bash
python3 -c "
from pathlib import Path
p = Path('backend/.env')
for key in ('CHAT_PROVIDER','EMBEDDING_PROVIDER','EMBEDDING_DIMENSIONS','GOOGLE_EMBEDDING_MODEL','GOOGLE_API_KEY'):
    for line in p.read_text().splitlines():
        if line.startswith(key + '='):
            v = line.split('=',1)[1].strip()
            print(f'{key}={\"set\" if key==\"GOOGLE_API_KEY\" else v} (len={len(v)})')
"
```

Expected before edits: `CHAT_PROVIDER=google`, `EMBEDDING_PROVIDER=ollama`, `EMBEDDING_DIMENSIONS=768`, `GOOGLE_EMBEDDING_MODEL=gemini-embedding-001`, `GOOGLE_API_KEY=set`.

- [ ] **Step 2: Set local embedding provider to Google**

In `backend/.env`, change:

```bash
EMBEDDING_PROVIDER=google
```

Leave `CHAT_PROVIDER=google`, `EMBEDDING_DIMENSIONS=768`, `GOOGLE_EMBEDDING_MODEL=gemini-embedding-001`, and the existing `GOOGLE_API_KEY` intact. Do not stage or commit this file.

- [ ] **Step 3: Update `.env.example` defaults**

In `backend/.env.example`, apply these exact values/comments:

```bash
# Default provider when the UI loads (users pick provider + model from live API catalogs)
# local | google | opencode
CHAT_PROVIDER=google

# --- Embeddings (Google AI Studio recommended) ---

# ollama | google | none
EMBEDDING_PROVIDER=google
EMBEDDING_DIMENSIONS=768
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

Keep the Google embeddings model line:

```bash
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
```

Update the Ollama comment so it is clearly optional for embeddings, not the recommended primary.

- [ ] **Step 4: Align Python settings default**

In `backend/app/config.py`, change:

```python
embedding_provider: EmbeddingProvider = "ollama"
```

to:

```python
embedding_provider: EmbeddingProvider = "google"
```

Leave `embedding_dimensions: int = 768` and `google_embedding_model: str = "gemini-embedding-001"` unchanged.

- [ ] **Step 5: Run config tests**

Run:

```bash
cd backend && uv run pytest tests/test_config.py -v
```

Expected: PASS. In particular, Google key is still required when `embedding_provider="google"`.

- [ ] **Step 6: Commit tracked defaults only**

```bash
git add backend/.env.example backend/app/config.py
git commit -m "$(cat <<'EOF'
chore: default chat and embeddings to Google AI Studio

EOF
)"
```

Do not add `backend/.env`.

---

### Task 2: Document Google as the production primary stack

**Files:**
- Modify: `backend/README.md`

**Interfaces:**
- Consumes: Task 1 defaults (`CHAT_PROVIDER=google`, `EMBEDDING_PROVIDER=google`, `gemini-embedding-001`, 768 dims)
- Produces: README production/chat/embedding sections that match those defaults

- [ ] **Step 1: Rewrite the production stack section**

Replace the current “Production stack (chat + retrieval)” block in `backend/README.md` with:

```markdown
### Production stack (chat + retrieval)

Production uses **Google AI Studio for chat and embeddings**. OpenCode and Ollama remain optional alternatives when configured.

| Layer | Provider | Config |
|-------|----------|--------|
| Chat LLM | Google AI Studio | `CHAT_PROVIDER=google`, `GOOGLE_API_KEY` |
| Query + document embeddings | Google (`gemini-embedding-001`) | `EMBEDDING_PROVIDER=google`, `GOOGLE_EMBEDDING_MODEL=gemini-embedding-001`, `EMBEDDING_DIMENSIONS=768` |
| Retrieval | Postgres `pgvector` + FTS | Hybrid search with RRF fusion; FTS fallback when embeddings fail |

**Retrieval flow:** user question → `/chat/stream` → Google agent → `search_filings` tool → Google `embedContent` (`RETRIEVAL_QUERY`) → vector search + Postgres FTS → fused passages → grounded answer with citations.

Stored filing chunks are embedded at ingest / re-embed time via Google into `document_chunks.embedding vector(768)`.
```

- [ ] **Step 2: Update the embeddings table and Google section**

In the Embeddings subsection:

1. Change the Ollama row from `` `ollama` (default) `` to `` `ollama` ``.
2. Change the Google row from `` `google` `` optional wording to `` `google` (default) ``.
3. Rename `#### Google embeddings (optional)` to `#### Google embeddings (default)`.
4. Keep the warning that query-time and stored embeddings must use the same model, and that switching providers requires `ingest.reembed`.

- [ ] **Step 3: Update the production rollout snippet**

Replace the production `.env` snippet under `#### Production rollout` with:

```bash
CHAT_PROVIDER=google
GOOGLE_API_KEY=...

EMBEDDING_PROVIDER=google
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSIONS=768
USE_OLLAMA=false
```

Keep any optional OpenCode / Ollama notes after that snippet if they already explain alternatives; do not present them as the primary stack.

- [ ] **Step 4: Keep the optional Ollama local-vs-production section accurate**

Leave the “Local vs production Ollama” section, but ensure surrounding text makes clear it applies only when `EMBEDDING_PROVIDER=ollama`, not to the default Google stack.

- [ ] **Step 5: Commit docs**

```bash
git add -f backend/README.md
git commit -m "$(cat <<'EOF'
docs: document Google AI Studio as primary chat and embeddings

EOF
)"
```

If `*.md` is gitignored, `-f` is required for the README add only when git refuses a normal add.

---

### Task 3: Re-embed the corpus with Google and verify

**Files:**
- Run: `backend/ingest/reembed.py` (no code changes expected)
- Verify via: `backend/tests/test_config.py`, `backend/tests/ingestion/test_embed.py`, `backend/tests/ingestion/test_reembed.py`, `backend/tests/ingestion/test_providers_google.py`

**Interfaces:**
- Consumes: `Settings.embedding_provider == "google"`, `Settings.google_api_key`, `Settings.google_embedding_model`, `Settings.embedding_dimensions`
- Produces: every `document_chunks.embedding` rewritten with Google `RETRIEVAL_DOCUMENT` vectors of length 768

- [ ] **Step 1: Confirm settings loaded for re-embed**

Run:

```bash
cd backend && uv run python -c "
from app.config import settings
print(settings.chat_provider)
print(settings.embedding_provider)
print(settings.embedding_dimensions)
print(settings.google_embedding_model)
print('google_key_set', bool(settings.google_api_key.strip()))
"
```

Expected:

```text
google
google
768
gemini-embedding-001
google_key_set True
```

- [ ] **Step 2: Count chunks before re-embed**

Run:

```bash
cd backend && uv run python -c "
from sqlalchemy import func, select
from app.database.document_chunk import DocumentChunk
from ingest.db import session_scope
with session_scope() as session:
    total = session.scalar(select(func.count()).select_from(DocumentChunk))
    print(f'chunks={total}')
"
```

Expected: `chunks=N` where `N >= 0`. If `N == 0`, skip Step 3 and note that there was nothing to re-embed.

- [ ] **Step 3: Run full re-embed**

Run:

```bash
cd backend && uv run python -m ingest.reembed
```

Expected stdout ending with:

```text
Re-embedded N/N chunks.
```

and exit code `0`.

If the run fails mid-way, do **not** flip traffic assumptions; re-run the same command until it reports `N/N` successfully. Partial batches already committed are Google vectors; a complete re-run makes the corpus consistent.

- [ ] **Step 4: Spot-check one stored embedding length**

Run only if Step 2 reported `N > 0`:

```bash
cd backend && uv run python -c "
from sqlalchemy import select
from app.database.document_chunk import DocumentChunk
from ingest.db import session_scope
with session_scope() as session:
    emb = session.execute(select(DocumentChunk.embedding).limit(1)).scalar_one()
    print(len(emb) if emb is not None else None)
"
```

Expected: `768`.

- [ ] **Step 5: Run focused non-integration tests**

Run:

```bash
cd backend && uv run pytest -m "not integration" \
  tests/test_config.py \
  tests/ingestion/test_embed.py \
  tests/ingestion/test_reembed.py \
  tests/ingestion/test_providers_google.py -v
```

Expected: PASS.

- [ ] **Step 6: No commit for DB mutation**

There is nothing to commit for the re-embed itself. Confirm `git status` does not stage `.env` or graphify artifacts.

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Set Google chat + embedding in local `.env` | Task 1 |
| Set same defaults in `.env.example` | Task 1 |
| Keep `gemini-embedding-001` + 768 dims | Tasks 1–3 |
| Update README primary stack | Task 2 |
| Run `uv run python -m ingest.reembed` | Task 3 |
| Verify `N/N` and focused tests | Task 3 |
| Keep OpenCode/Ollama optional | Tasks 1–2 |
| Do not commit secrets | Tasks 1 and 3 |

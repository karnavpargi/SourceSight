# SourceSight Backend

FastAPI service for SourceSight. Run commands from this `backend/` folder.

## Setup

```bash
uv sync
cp .env.example .env
```

Fill `.env` with Supabase, Postgres, chat provider keys, and `ALLOWED_ORIGINS`. `DATABASE_URL` should be the direct Supabase database URL, not the pooler URL.

## Run The API

```bash
uv run uvicorn app.main:app --reload
```

Open the API docs at `http://localhost:8000/docs`.

### HTTP/2 (optional, inbound)

Outbound calls (OpenCode, Google, Ollama, Supabase) use `httpx` directly in each provider module.

To serve the API itself over HTTP/2 locally you need **Hypercorn** with **TLS** (browsers require HTTPS for HTTP/2):

```bash
# one-time self-signed cert for local dev
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"

uv run hypercorn app.main:app --bind 0.0.0.0:8000 --certfile cert.pem --keyfile key.pem --reload
```

Then open `https://localhost:8000/docs` (accept the browser security warning).

Plain HTTP without TLS stays HTTP/1.1 — there is no `--h2` flag:

```bash
uv run hypercorn app.main:app --bind 0.0.0.0:8000 --reload
```

For day-to-day dev, `uvicorn` on `http://localhost:8000` is enough.

## Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"ok": true}
```

## Common Commands

```bash
uv run ruff check .
uv run pytest
uv run pytest tests/ingestion --cov=ingest --cov-fail-under=100
uv run alembic upgrade head
```

The ingestion pipeline has **100% line coverage** (`tests/ingestion/`, 101 tests). Run the third command after changing anything under `ingest/`.

Use `app.config.settings` for environment configuration. Do not read environment variables directly in app code.

## Chat LLM providers

`CHAT_PROVIDER` in `.env` is only the default tab when the UI loads. **Model names are never stored in config** — they are fetched live on every catalog/validation request.

| Provider | `CHAT_PROVIDER` | Required env | Live model catalog |
|----------|-----------------|--------------|-------------------|
| Local (Ollama) | `local` | `OLLAMA_BASE_URL` | `{OLLAMA_BASE_URL}/api/tags` |
| Google AI Studio | `google` | `GOOGLE_API_KEY` | [generativelanguage.googleapis.com/v1beta/models](https://generativelanguage.googleapis.com/v1beta/models) |
| OpenCode Zen Go | `opencode` | `OPENCODE_API_KEY`, `OPENCODE_BASE_URL` | [opencode.ai/zen/go/v1/models](https://opencode.ai/zen/go/v1/models) |

`CHAT_PROVIDER=ollama` is accepted as an alias for `local`.

Users pick provider and model in the chat UI. The backend exposes `GET /chat/providers` (live catalogs) and requires `provider` + `model` on `POST /chat/stream`.

### Multi-model routing (cost optimization)

The backend uses an **extract-first, escalate-only** pipeline:

- **Router + extractor model** (cheap): `CHAT_ROUTER_MODEL` (validated against Google’s live catalog).
- **Synthesis model** (per request): the `provider`/`model` selected by the client (often the UI default `CHAT_MODEL`).

Per-turn limits (design):

- **Calls**: 1 router, 1 extractor, 0–1 synthesis, 0–1 citation correction (no full agent retries).
- **Retrieval**:
  - standard: ≤3 queries, 5 hits/query, 8 unique passages
  - broad: ≤5 queries, 5 hits/query, 15 unique passages

Configuration:

```dotenv
CHAT_ROUTER_MODEL=gemini-flash-lite-latest
CHAT_MODEL=gemini-3.5-flash-lite
# Paid-tier USD per 1M tokens [input, output]
CHAT_MODEL_PRICES={"gemini-flash-lite-latest":[0.30,2.50],"gemini-3.5-flash-lite":[0.30,2.50]}
```

If `CHAT_MODEL_PRICES` contains exact model IDs for every model used in a turn, logs include `estimated_cost_usd`. Otherwise it is `null`.

### Production stack (chat + retrieval)

Production uses **Google AI Studio for chat and embeddings**. OpenCode and Ollama remain optional alternatives when configured.

| Layer | Provider | Config |
|-------|----------|--------|
| Chat LLM | Google AI Studio | `CHAT_PROVIDER=google`, `GOOGLE_API_KEY` |
| Query + document embeddings | Google (`gemini-embedding-001`) | `EMBEDDING_PROVIDER=google`, `GOOGLE_EMBEDDING_MODEL=gemini-embedding-001`, `EMBEDDING_DIMENSIONS=768` |
| Retrieval | Postgres `pgvector` + FTS | Hybrid search with RRF fusion; FTS fallback when embeddings fail |

**Retrieval flow:** user question → `/chat/stream` → Google agent → `search_filings` tool → Google `embedContent` (`RETRIEVAL_QUERY`) → vector search + Postgres FTS → fused passages → grounded answer with citations.

Stored filing chunks are embedded at ingest / re-embed time via Google into `document_chunks.embedding vector(768)`.

### Local vs production Ollama

This section applies only when `EMBEDDING_PROVIDER=ollama`, not the default Google stack.

Only `OLLAMA_BASE_URL` changes between environments. Keep the same `OLLAMA_EMBEDDING_MODEL` (`nomic-embed-text`) so ingest-time and query-time vectors stay in the same space.

| Environment | `OLLAMA_BASE_URL` | Typical use |
|-------------|-------------------|-------------|
| Local dev | `http://localhost:11434` | Run `ollama serve` on your machine; ingest, re-embed, and API query embeddings all hit local Ollama |
| Production (OCI) | `https://ollama.kpargi.eu.org` | API and batch jobs on the Pi/OCI host call the remote Ollama server (HTTPS via Cloudflare Tunnel) |

Local setup:

```bash
brew services start ollama   # or: ollama serve
ollama pull nomic-embed-text
```

Production `.env` snippet:

```bash
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=https://ollama.kpargi.eu.org
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768
```

You can ingest or re-embed on your laptop against local Ollama, then deploy the API with the remote URL — same model, compatible vectors.

## Ingestion Pipeline

One-off scripts in `ingest/` load the local SEC 10-K corpus (`data/downloads/`) into Supabase as chunked, embedded, search-ready rows.

| Module | Purpose |
|--------|---------|
| `ingest/extract.py` | SEC HTML → clean Markdown |
| `ingest/chunk.py` | Section-aware chunker (~800 tokens, ~150 overlap) |
| `ingest/embed.py` | Embedding provider routing (Ollama or Google Gemini) |
| `ingest/load.py` | Writes `source_documents` + `document_chunks` |
| `ingest/reembed.py` | Re-embed existing chunks after switching providers |
| `ingest/run.py` | Orchestrator — walks `data/downloads/manifest.json` |

Re-runs are idempotent: filings already in the DB (matched by `ticker`, `form_type`, `fiscal_year`, `accession_number`) are skipped.

### Embeddings

Retrieval uses hybrid **vector + full-text search**. Query-time and ingestion embeddings must use the **same model** — you cannot mix `nomic-embed-text` vectors with Google `gemini-embedding-001`. If you switch embedding providers, run `ingest.reembed` before serving traffic.

| `EMBEDDING_PROVIDER` | Use case | Required env |
|----------------------|----------|--------------|
| `ollama` | Local or remote Ollama (`/api/embed`) | `OLLAMA_BASE_URL`, `OLLAMA_EMBEDDING_MODEL=nomic-embed-text` |
| `google` (default) | Gemini via Google AI Studio | `GOOGLE_API_KEY`, `GOOGLE_EMBEDDING_MODEL=gemini-embedding-001` |
| `none` | FTS-only retrieval (no vector search) | — |

`EMBEDDING_DIMENSIONS` must match the DB column (`vector(768)`).

`USE_OLLAMA` controls whether the **local Ollama chat provider** appears in `/chat/providers`; it does not gate embeddings. Use `EMBEDDING_PROVIDER` for vector search.

#### Google embeddings (default)

If you use `EMBEDDING_PROVIDER=google`, the backend calls native `embedContent` / `batchEmbedContents` (not `/v1/embeddings`). Task types: `RETRIEVAL_QUERY` for chat queries, `RETRIEVAL_DOCUMENT` for filing chunks. Requires a full re-embed when switching from Ollama.

#### Production rollout

Set in `.env` on the production host:

```bash
CHAT_PROVIDER=google
GOOGLE_API_KEY=...

EMBEDDING_PROVIDER=google
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSIONS=768
USE_OLLAMA=false
```

For chat via OpenCode or embeddings via Ollama instead, set `CHAT_PROVIDER=opencode` / `EMBEDDING_PROVIDER=ollama` and the env vars listed in the provider tables above (see **Local vs production Ollama** when using Ollama embeddings).

If the corpus was embedded with a different provider or model, **re-embed before promoting**:

```bash
uv run python -m ingest.reembed
```

Confirm every chunk has a vector, then restart the API:

```bash
uv run python -c "
from sqlalchemy import text
from ingest.db import get_engine
with get_engine().connect() as conn:
    total = conn.execute(text('select count(*) from document_chunks')).scalar()
    embedded = conn.execute(text('select count(*) from document_chunks where embedding is not null')).scalar()
    print(f'embedded: {embedded}/{total}')
"
```

Smoke-check: ask a representative question via `/chat/stream` and confirm the backend does **not** log `retrieval.embedding_unavailable`. FTS fallback remains enabled when Ollama is unreachable.

**Fresh ingest** (new corpus):

```bash
# local Ollama (default .env.example)
ollama pull nomic-embed-text
uv run python -m ingest.run

# remote Ollama (e.g. production host ingesting directly)
OLLAMA_BASE_URL=https://ollama.kpargi.eu.org uv run python -m ingest.run
```

**Re-embed existing chunks** after switching providers (updates `document_chunks.embedding` in place in stable `document_id, chunk_index` order; safe for citations and chat history; commits per batch so failures are resumable):

```bash
uv run python -m ingest.reembed
# optional: --batch-size 32 --document-id <uuid>
```

Expected output for the bundled corpus: **25 filings**, **~4,100 chunks** (varies by filing length).

### Verify ingest

```bash
uv run python -c "
from sqlalchemy import text
from ingest.db import get_engine
with get_engine().connect() as conn:
    print('docs:', conn.execute(text('select count(*) from source_documents')).scalar())
    print('chunks:', conn.execute(text('select count(*) from document_chunks')).scalar())
"
```

Spot-check full-text search (Apple supply-chain risk):

```sql
select section, left(content, 200)
from document_chunks
where search_vector @@ plainto_tsquery('english', 'supply chain concentration')
  and metadata->>'ticker' = 'AAPL'
limit 3;
```

### Schema and migrations

SQLAlchemy models in `app/database/` are the source of truth. After model changes:

```bash
uv run alembic revision --autogenerate -m "describe change"   # review output
uv run alembic upgrade head
```

`document_chunks.embedding` is `vector(768)` for `nomic-embed-text` (and `gemini-embedding-001` if using Google). `search_vector` is a Postgres generated column (`to_tsvector('english', content)`) — do not insert into it directly.

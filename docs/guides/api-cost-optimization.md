# API cost optimization (server-owned evidence)

**Date:** 2026-07-24  
**Status:** Approved  
**Target:** Under ₹1 per answer (same answer quality and grounding contract)

## Problem (measured)

One Amazon AWS vs segment comparison turn cost **₹8.13** (₹38.05 → ₹46.18 → ₹53.07).

Observed for that turn:

| Metric | Value |
|---|---|
| Agent runs | 2 (initial + grounding retry) |
| Gemini `streamGenerateContent` calls | 10 |
| Embedding calls | 5 |
| Unique passages | 17 |
| Tokens run 1 | 77,809 in / 13,425 out |
| Tokens run 2 | 90,214 in / 14,233 out |
| Total | ~168k in / ~28k out |
| Visible answer length | ~2.3k characters |

Root causes:

1. **Over-fetching** — up to 10 chunks per search (~800 tokens each), multiple searches, surrounding chunks.
2. **Expensive structured output** — model regenerates full `cited_passages` / passage objects as output tokens.
3. **Full grounding retry** — failed citation validation restarts the whole agent (retrieval + generation again).
4. **No Gemini context caching** — not the primary driver here (see Out of scope).

Not a driver: full chat history is already not sent to the model (latest user text only).

## Goal

Reduce cost to **under ₹1** for this question class **without reducing visible answer quality**. Same grounded answer shape for the frontend: answer text, citations, cited passages.

## Approach

**Server-owned evidence pipeline** (chosen over config-only tweaks and precomputed financial facts).

### Architecture

1. Agent may run at most **3 focused searches** via a **batched retrieval tool**.
2. Each search returns at most **5 hits**; the turn keeps at most **8 unique passages**.
3. Tools return a **compact evidence** shape: evidence alias (`E1`…`E8`), content, ticker, fiscal year, section. No UUID / filing metadata dump in model context.
4. Model outputs a **`GroundedDraft`**: answer text + citations that reference evidence aliases and short excerpts. Model never emits full `SourcePassage` objects or chunk UUIDs.
5. Backend **finalizer** maps aliases → trusted `SourcePassage` records from the turn evidence registry, builds UUID citations, and hydrates `cited_passages` for the existing API/UI contract.
6. Generation has a hard **output budget (~1,000–2,000 tokens)** and a bounded model-request / tool-call budget.
7. On grounding failure: **deterministic repair first**; if needed, **one compact citation-only correction** using already-retrieved evidence and **no retrieval tools**. Second failure → existing refusal message.
8. Every turn logs cumulative model calls, embedding calls, passages, input/output tokens, correction usage.

### Components

| Component | Responsibility |
|---|---|
| Compact evidence model | Alias + text + ticker/year/section only |
| Batched retrieval tool | Up to 3 queries, dedupe, cap 8 passages |
| Turn evidence registry | Maps `E1`–`E8` → server-side `SourcePassage` |
| Grounded draft schema | Answer + alias citations + short excerpts |
| Server finalizer | Alias → UUID citations + hydrate passages |
| Turn budget | Cap model requests, searches, passages, corrections, output tokens |
| Usage recorder | Aggregate tokens/calls across corrections |

### Safeguards

- Unknown evidence aliases are rejected.
- Citation correction cannot call retrieval tools.
- Search-budget exhaustion → best partial grounded answer or refusal.
- Auth, persistence, streaming, and frontend wire format stay unchanged.

## Success criteria (same Amazon question)

- Gemini generation calls ≤ 4 (including optional correction)
- Embedding calls ≤ 3
- Unique passages ≤ 8
- Total input tokens ≤ 25,000
- Total output tokens ≤ 2,000
- Cost under ₹1
- Frontend still receives answer + citations + cited passages

## Testing

1. Unit: alias → chunk UUID mapping; finalizer hydrates passages.
2. Unit: batched search dedupes and enforces 8-passage cap.
3. Unit: grounding failure → one no-retrieval correction; second failure refuses.
4. Unit: draft schema rejects UUIDs / full `SourcePassage` objects.
5. Orchestrator: usage log includes cumulative tokens across correction.
6. Regression: grounding validator still rejects uncited numbers and invented chunk IDs.
7. Manual: rerun measured Amazon question; compare spend vs ₹8.13 baseline.

## Out of scope

- **Gemini context caching** — can help a little for a large *stable* prefix, but most billed tokens here are dynamic retrieval + tool transcripts + structured output. Caching alone will not hit the ₹1 target. Revisit later if instructions/static corpus prefixes grow large enough for cache minimums.
- Precomputed financial fact tables
- Frontend UI redesign
- Chat history changes (already not sent)

## Expected budget after change

Typical turn: roughly **10k–25k input** and **under 2k output**, targeting **₹0.50–₹1** on current `gemini-3.5-flash-lite` pricing for this question class.

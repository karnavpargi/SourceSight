# Multi-model cost routing for SourceSight

**Date:** 2026-07-25  
**Status:** Approved

## Goal

Keep the server-owned evidence pipeline on `feat/api-cost-optimization`, then add an extract-first, escalate-only model flow. A low-cost Gemini Flash-Lite model plans retrieval and extracts grounded facts. The user-selected model performs narrative synthesis only when the question requires it.

The design must support all ten analyst question patterns in `docs/client-brief.md`, not only the Amazon cost benchmark.

## Current state

The optimization branch already:

- batches retrieval queries;
- caps search results and unique evidence;
- sends compact `E1`…`E8` aliases instead of full source metadata;
- has the model emit `GroundedDraft` rather than trusted UUIDs and passages;
- resolves aliases to source passages on the server;
- replaces full grounding re-retrieval with one citation-only correction;
- records cumulative turn usage; and
- passes the non-integration suite.

The first live measurement reduced retrieved passages from 17 to 8 and input tokens from roughly 168,000 to 81,000, but it still ended in a grounding refusal and missed the 25,000-input-token target. Later truncation and retry changes have not yet been remeasured because provider quota blocked the run.

## Scope

### In scope

- A cheap structured query router and retrieval planner.
- Batched retrieval driven by a validated plan.
- Cheap structured extraction of numerical and qualitative facts.
- Deterministic routing between a cheap final draft and a synthesis model.
- Adaptive but bounded retrieval profiles for broad comparisons.
- Existing server-owned evidence aliases and deterministic grounding.
- Per-stage token, call, passage, route, and estimated-cost telemetry.
- Regression evaluation against all ten client-brief questions.

### Out of scope

- Pre-extracting financial tables during ingestion.
- Gemini explicit context caching.
- Adding OpenAI as a new provider solely for routing.
- Model-generated SQL.
- Sending prior chat history to either model.
- A second LLM-based grounding auditor.
- Unbounded retries or retrieval expansion.

## Model policy

The server has two model roles:

- **Router/extractor model:** configured by `CHAT_ROUTER_MODEL`, intended to be the lowest-cost tool-capable Flash-Lite model available to the deployment. `gemini-2.0-flash-lite` may be configured while it remains available; operators can replace it with a current Flash-Lite model without code changes.
- **Synthesis model:** the Google model selected for the request, normally `gemini-3.5-flash-lite`. Existing model selection therefore remains useful as a manual choice of synthesis model.

The router model is validated against Google's live model catalog. If it is unavailable, malformed, or fails, the turn falls back to the selected synthesis model with the default bounded retrieval profile. The application does not infer model price from a model name and does not silently select an arbitrary model.

## Question classes

Routing is semantic and must not hard-code the numbered examples.

### Extractive

Questions primarily asking for disclosed values, trends, or tables. Client-brief examples include revenue mix, segment operating income, revenue trends, capital expenditure, and geographic exposure.

The router/extractor model can produce the final grounded draft when coverage is sufficient and no material inference is requested.

### Synthesis

Questions requiring comparison of narrative language, changes in disclosure, or reasoning across companies and years. Examples include demand drivers, infrastructure constraints, changed risk language, and supplier urgency.

The router/extractor model prepares the plan and grounded fact set. The synthesis model receives only that compact material and supporting excerpts, never the tool transcript.

### Boundary or refusal

Questions asking the filings to prove a causal claim that the corpus may not establish. The route still retrieves relevant evidence, but both prompts and deterministic checks prefer an explicit limitation or refusal over unsupported inference.

## Components

### `QueryPlan`

A strict structured response from the router containing:

- route class: `extractive`, `synthesis`, or `boundary`;
- companies or tickers;
- fiscal years;
- requested metrics, segments, and themes;
- one to three primary retrieval queries;
- up to two reserve queries for a single coverage expansion;
- required coverage dimensions; and
- whether narrative synthesis is required.

The plan has no SQL field and cannot name arbitrary database filters. The backend validates query count, lengths, known tickers, fiscal-year ranges, and route values before retrieval.

### Query router

The router receives only the current user question and a compact description of corpus coverage. It has no tools and a small output budget, targeted at 300 tokens. It makes exactly one attempt.

### Batched retriever

The backend executes the validated plan using existing hybrid vector and full-text retrieval. It does not execute model-generated SQL.

Two profiles are available:

- standard: at most 3 queries, 5 hits per query, and 8 unique passages;
- broad comparison: at most 5 queries, 5 hits per query, and 15 unique passages.

The broad profile is allowed only when the validated plan spans multiple companies or requires coverage that cannot fit the standard profile. Retrieval remains deduplicated by chunk UUID. After primary retrieval, the backend compares evidence ticker/year metadata with the plan's required coverage. It may execute the reserve queries once, within the same total query and passage caps, before fact extraction.

### Structured fact extractor

The router/extractor model receives compact evidence and emits typed facts:

- company and ticker;
- fiscal year or period;
- metric, segment, or theme;
- verbatim disclosed value and unit when numerical;
- short qualitative finding when narrative;
- evidence alias; and
- coverage status: `supported`, `missing`, or `conflicting`.

For an `extractive` route, the same structured response also contains a `GroundedDraft`; synthesis and boundary routes omit it. This avoids a third cheap-model call merely to format an extractive answer.

The extractor must not estimate missing values. Deterministic validation confirms that each alias exists and that normalized numerical values appear in the cited source content. Unsupported facts are discarded before the optional draft is finalized or facts are passed to synthesis.

### Decision gate

The backend decides the next step without another classifier call:

- `extractive` + sufficient coverage: use the `GroundedDraft` included in the validated extraction response;
- `synthesis`: call the selected synthesis model with the question, validated facts, missing-coverage notes, and compact supporting excerpts;
- `boundary`: call the selected synthesis model only when needed to explain the evidence boundary; otherwise return a grounded limitation;
- missing coverage after extraction: qualify the answer or refuse; retrieval expansion has already been decided from evidence metadata before this stage.

### Grounded finalizer

The existing finalizer remains authoritative:

- resolve evidence aliases to server-held `SourcePassage` objects;
- hydrate UUID citations and cited passages;
- run deterministic repair and validation;
- permit one citation-only correction with retrieval disabled; and
- refuse after a second grounding failure.

## End-to-end flow

1. Receive the latest user question and selected synthesis model.
2. Run the cheap router once and validate `QueryPlan`.
3. Execute the primary batched retrieval operation under the selected budget profile.
4. Compare retrieved ticker/year metadata with required coverage and, if needed, execute reserve queries once within the same caps.
5. Register compact evidence aliases and run cheap fact extraction.
6. Validate aliases, numerical values, and coverage.
7. Use the extractive draft from the cheap model, or escalate compact facts and excerpts to the synthesis model.
8. Finalize aliases into trusted citations and validate grounding.
9. If necessary, run one no-retrieval citation correction.
10. Persist and stream the unchanged frontend wire format.

No stage receives previous chat turns. The synthesis model does not receive router messages, extraction prompts, database objects, or the complete PydanticAI tool history.

## Budgets

Per question:

- router calls: 1;
- extraction calls: 1;
- synthesis calls: 0 or 1;
- citation corrections: 0 or 1;
- retrieval expansion: 0 or 1, within the active query and passage caps;
- full agent re-runs: 0.

Stage output caps are:

- router: 300 tokens;
- standard extraction, including an optional draft: 2,800 tokens;
- broad extraction, including an optional draft: 3,500 tokens;
- final draft or synthesis: 2,800 tokens; and
- citation correction: 1,200 tokens.

The existing turn budget is extended with these stage-specific limits rather than replaced with unbounded configuration.

## Failure handling

- Router model unavailable: log the failure and use the selected synthesis model with standard bounded retrieval.
- Invalid router output: reject it and use the same fallback; never partially trust invalid fields.
- Unknown ticker or year outside corpus coverage: mark the missing scope and continue only with known coverage.
- Conflicting numerical facts: retain both cited values with their periods or refuse to reconcile them.
- Missing numerical facts: state the gap; never interpolate.
- Extraction failure: use the selected synthesis model over compact evidence, not raw tool history.
- Synthesis failure: return the existing provider-unavailable response.
- Citation failure: one no-retrieval correction, then the existing grounded refusal.

## Telemetry

One turn log records:

- route class and budget profile;
- router, extraction, synthesis, and correction model IDs;
- calls and input/output tokens per stage;
- embedding calls, retrieval queries, and unique passages;
- retrieval expansion and correction use;
- grounding outcome;
- total latency; and
- estimated cost when a configured server-side price table contains the exact model ID.

Unknown prices produce `estimated_cost=None`; they do not silently use another model's rate. Provider billing remains the source of truth for acceptance measurement.

## Testing

### Unit and integration tests

- `QueryPlan` validation and malformed-output fallback.
- Semantic routing fixtures covering all ten client-brief questions.
- Standard and broad budget enforcement.
- Deduplication and focused retrieval expansion.
- Fact extraction with currency, percentages, negative values, units, and missing data.
- Deterministic numerical-value-to-evidence checks.
- Extractive completion without synthesis.
- Synthesis escalation with compact context only.
- Boundary answers and clean refusal.
- Router-model unavailability fallback.
- User-selected synthesis model preservation.
- Alias-to-UUID finalization and invented-alias rejection.
- Cumulative per-stage usage telemetry.
- Existing grounding adversarial tests and the full non-integration suite.

### Live evaluation

Run all ten questions in `docs/client-brief.md` against the same indexed corpus and record route, coverage, answer/refusal, citations, stage tokens, and provider-reported spend. Compare the Amazon question directly with the ₹8.13 baseline.

## Acceptance criteria

- Amazon benchmark costs under ₹1 and uses no more than 25,000 input tokens.
- The ten-question suite has median cost under ₹1.
- No normally answered question exceeds ₹2 without a logged budget warning.
- Every factual answer has valid source citations, or the turn cleanly refuses.
- No numerical claim survives validation unless its normalized value is present in cited evidence.
- Router and extractor never receive prior chat history.
- Synthesis receives only validated facts and compact excerpts.
- No full agent grounding retry occurs.
- Existing frontend answer, citation, and cited-passage shapes remain unchanged.
- The full non-integration suite remains green.

## Deferred follow-up

If the ten-question evaluation still misses cost or coverage targets, the next design should add ingestion-time extraction of standardized financial tables. Context caching should be reconsidered only when telemetry shows a large, stable prompt prefix reused often enough to offset cache storage cost.

# Task 4 – Batched compact search tool + agent output type

## Summary

Implemented a batched `search_filings(queries: list[str])` tool that returns compact evidence aliases, wired it through a new `DocumentRetriever.search_filings_batch` API and an enriched `DocumentAgentDeps`, and switched the document agent output type from `GroundedAnswer` to `GroundedDraft` with alias-based citations. Updated agent instructions, dependencies, and tests to reflect the new compact evidence workflow while keeping retrieval and recording behavior consistent.

## Code Changes

- **Assistant dependencies (`backend/app/assistant/deps.py`)**
  - Extended `DocumentRetriever` protocol with `search_filings_batch(queries: list[str], *, limit_per_query: int = 5) -> list[SourcePassage]` while preserving existing `search_filings`, `read_chunk`, and `read_surrounding_chunks` signatures.
  - Enriched `DocumentAgentDeps` to include `evidence: EvidenceRegistry`, `usage: TurnUsage`, `budget: TurnBudget = DEFAULT_TURN_BUDGET`, and `search_count: int = 0`, in addition to existing `user_id`, `thread_id`, `retriever`, and `grounding_validator`.
- **Retriever implementation (`backend/app/retrieval/document_retriever.py`)**
  - Implemented `SessionDocumentRetriever.search_filings_batch` to iterate over queries, call `search_filings` for each with `limit_per_query`, and deduplicate `SourcePassage` results by `chunk_id` via a `dict[UUID, SourcePassage]`.
  - Added `SessionPerCallDocumentRetriever.search_filings_batch` that opens a fresh DB session per call and delegates to `SessionDocumentRetriever.search_filings_batch`, mirroring the pattern used for other methods.
- **Agent behavior (`backend/app/assistant/agent.py`)**
  - Switched `document_agent` to `output_type=PromptedOutput(GroundedDraft)` and imported `CompactEvidence`/`GroundedDraft` instead of `GroundedAnswer`.
  - Introduced `_search_filings_impl(deps: DocumentAgentDeps, queries: list[str]) -> list[CompactEvidence]` as the core implementation:
    - Enforces `TurnBudget.max_searches` using `deps.search_count`.
    - Cleans and truncates the `queries` list (strip whitespace, drop empties, cap to `max_searches`).
    - Calls `deps.retriever.search_filings_batch` with `limit_per_query=budget.max_hits_per_search`.
    - Registers passages in the `EvidenceRegistry`, returning only newly created `CompactEvidence` rows (aliases like `E1`, `E2`) and never exposing `chunk_id`.
    - Updates `TurnUsage` via `deps.usage.record_passages(len(deps.evidence.all_passages()))`.
  - Replaced the old single-query `search_filings` tool plus `read_chunk` / `read_surrounding_chunks` tools with a single batched tool:
    ```python
    @document_agent.tool
    def search_filings(
        ctx: RunContext[DocumentAgentDeps],
        queries: list[str],
    ) -> list[CompactEvidence]:
        return _search_filings_impl(ctx.deps, queries)
    ```
- **Instructions (`backend/app/assistant/instructions.md`)**
  - Updated **Tools** section to describe `search_filings(queries: list[str])` returning compact evidence with aliases like `E1`, removed references to `read_chunk` and `read_surrounding_chunks`, and documented the multi-query pattern for cross-company/year comparisons.
  - Revised **Output contract** to:
    - Refer to `GroundedDraft` (two fields: `answer` and `citations`).
    - Define `citations` in terms of `DraftCitation` using `evidence_alias` (e.g. `E1`) instead of `chunk_id`, with the server resolving aliases to `chunk_id`/`SourcePassage` after the draft.
  - Adjusted **Workflow** to use `search_filings` with 1–3 focused queries within the budget and to build a `GroundedDraft` with markers and alias-based citations.
- **Assistant tests (`backend/tests/assistant/test_agent.py`)**
  - `test_load_instructions_encodes_grounding_contract`: now asserts that instructions mention `GroundedDraft`, `search_filings`, `queries`, an example alias like `E1`, and that `read_chunk` / `read_surrounding_chunks` are absent while still referencing “stock recommendation”.
  - `test_document_agent_registers_retrieval_tools`: updated to expect only `["search_filings"]` in `document_agent._function_toolset.tools`.
  - `test_document_agent_run_invokes_search_filings_tool`: adjusted the `FunctionModel` mock to:
    - Emit `ToolCallPart("search_filings", {"queries": ["AMZN AWS operating income 2024"]})` on the first step.
    - Return a `GroundedDraft`-shaped JSON payload (`{"answer": "...", "citations": []}`) on the second step.
    - Assert that `StubRetriever.last_query` is `"AMZN AWS operating income 2024"` and that `result.output` is a `GroundedDraft` with an empty `citations` list.
  - Added `_sample_passage()` helper to construct a realistic `SourcePassage` for tests.
  - New test `test_search_filings_tool_returns_compact_aliases_without_chunk_id`:
    - Uses a `StubBatchRetriever` with `search_filings_batch` returning a single `_sample_passage()`.
    - Constructs a lightweight `deps` `SimpleNamespace` with `budget=DEFAULT_TURN_BUDGET`, real `EvidenceRegistry`, `TurnUsage`, and the stub retriever.
    - Calls `_search_filings_impl(deps, ["AMZN AWS operating income 2024"])` and asserts:
      - The result is a list of `CompactEvidence`.
      - Serialized rows from `model_dump()` lack any `chunk_id` field.
      - At least one alias starts with `"E"`.
- **Assistant deps tests (`backend/tests/assistant/test_deps.py`)**
  - Extended `StubRetriever` with a `search_filings_batch` stub that records a joined query and returns an empty `list[SourcePassage]`.
  - Updated `test_document_agent_deps_wires_runtime_services` to:
    - Construct `EvidenceRegistry`, `TurnUsage`, and `TurnBudget`.
    - Build `DocumentAgentDeps` with the new `evidence`, `usage`, and `budget` fields.
    - Assert the new fields are wired correctly and that `search_count` defaults to `0`.
- **Plan document**
  - Added `docs/superpowers/plans/2026-07-24-task-4-batch-compact-search.md` describing the step-by-step implementation and TDD flow for this task (tool surface, deps, retriever, tests, and commit steps).

## Tests Run

All commands run from `backend/` in the api-cost-optimization worktree:

- **Focused assistant tests**
  - `uv run pytest tests/assistant/test_agent.py tests/assistant/test_deps.py -v`
- **Full assistant suite**
  - `uv run pytest tests/assistant/ -v`
- **Recording retriever**
  - `uv run pytest tests/chat/test_recording_retriever.py -v`
- **Retriever**
  - `uv run pytest tests/retrieval/test_document_retriever.py -v`
- **Full backend suite (sanity check)**
  - `uv run pytest -q`  
    - Result: `1 failed, 273 passed, 3 skipped, 1 error`  
    - **Failure:** `tests/ingestion/test_extract.py::test_extract_main_usage_error` – `FileNotFoundError: '-q'` when `ingest.extract` is invoked as `__main__` under `pytest -q` (CLI argument leaking into `sys.argv[1]`).  
    - **Error:** `tests/chat/test_smoke_integration.py::test_client_brief_aws_question_returns_cited_answer` – missing `ingested_corpus` fixture in this environment. These appear to be existing harness/environment issues rather than regressions from Task 4, and the task’s target assistant/retriever suites are all green.

## Concerns / Follow-Ups

- **Orchestrator finalize glue (Task 5):** The document agent now emits a `GroundedDraft`; orchestration and correction paths that currently expect `GroundedAnswer` will need to resolve `DraftCitation.evidence_alias` via `EvidenceRegistry` and produce the final `GroundedAnswer` + `cited_passages` in the orchestrator layer (explicitly deferred to Task 5 per brief).
- **Recording retriever compatibility:** `_RecordingRetriever` continues to rely only on `search_filings`, `read_chunk`, and `read_surrounding_chunks` and remains green in its tests; if future tasks route the agent through the recorder with `search_filings_batch`, it may be worth adding a thin `search_filings_batch` implementation that delegates to repeated `search_filings`.
- **Global pytest invocation:** Running the entire backend test suite with `pytest -q` currently exposes the ingestion CLI argument coupling and missing `ingested_corpus` fixture; these are out of scope for Task 4 but should be addressed separately if the project expects `pytest` to be green in all environments.


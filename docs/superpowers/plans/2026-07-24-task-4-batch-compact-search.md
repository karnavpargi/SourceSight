# Task 4: Batched compact search Implementation Plan
> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a batched compact search tool for the document assistant, wire it through a new `DocumentRetriever.search_filings_batch` API and evidence registry, and switch the document agent to emit `GroundedDraft` instead of `GroundedAnswer`.

**Architecture:** Extend the retrieval layer with a batch search helper that deduplicates passages by `chunk_id`, introduce richer per-turn dependencies (`EvidenceRegistry`, `TurnBudget`, `TurnUsage`, `search_count`) into `DocumentAgentDeps`, expose a single `search_filings(queries: list[str])` tool that populates compact evidence aliases and respects the budget, and update the agent output type and instructions to speak in terms of aliases (`E1`, `E2`, …) and `GroundedDraft`. Tests drive the changes at the assistant, deps, and retriever-recording layers.

**Tech Stack:** Python 3, PydanticAI, FastAPI backend, SQLAlchemy sessions for retrieval, pytest.

## Global Constraints

- Follow existing backend style and dependency policy from `AGENTS.md`.
- No new runtime dependencies unless strictly necessary (not expected here).
- Keep `DocumentRetriever` protocol backwards-compatible for existing orchestrator usages; add `search_filings_batch` without removing `search_filings`, `read_chunk`, or `read_surrounding_chunks` from the protocol.
- Respect `TurnBudget` defaults (`max_searches=3`, `max_hits_per_search=5`, `max_unique_passages=8`, etc.).
- `EvidenceRegistry.register()` must not re-emit duplicates; it only returns compact rows for newly registered passages.
- Do not implement orchestrator correction path beyond minimal glue needed to keep tests green.

---

### Task 1: Update assistant tests for new tools and output type

**Files:**
- Modify: `backend/tests/assistant/test_agent.py`
- Modify: `backend/tests/assistant/test_deps.py`

**Interfaces:**
- Consumes: Existing `DocumentAgentDeps`, `document_agent`, `GroundedAnswer`, `GroundedDraft`, `EvidenceRegistry`.
- Produces: Expectations that `document_agent` registers only a `search_filings` tool, that instructions mention `GroundedDraft` / aliases / multi-query `search_filings`, and that the agent’s `FunctionModel` flow emits a `GroundedDraft` JSON payload.

- [ ] **Step 1: Adjust retrieval tool registration test**

```python
def test_document_agent_registers_retrieval_tools() -> None:
    tool_names = sorted(document_agent._function_toolset.tools.keys())
    assert tool_names == ["search_filings"]
```

- [ ] **Step 2: Update instructions grounding contract test**

Change `test_load_instructions_encodes_grounding_contract` to assert that:

- `"GroundedDraft"` appears instead of (or in addition to) `"GroundedAnswer"`.
- `"search_filings"` is still present but mentions multiple `queries`.
- Aliases like `"E1"` and `[n]` markers are referenced.
- `read_chunk` / `read_surrounding_chunks` are **not** mentioned.

- [ ] **Step 3: Update FunctionModel test to use `GroundedDraft` and batched queries**

Modify `test_document_agent_run_invokes_search_filings_tool` so that:

- The first `ModelResponse` uses `ToolCallPart("search_filings", {"queries": ["AMZN AWS operating income 2024"]})`.
- The second `ModelResponse` returns JSON for a `GroundedDraft`:

```python
{
    "answer": "This corpus does not contain enough evidence to answer that.",
    "citations": [],
}
```

- The final assertion compares `result.output` to `GroundedDraft(...)` instead of `GroundedAnswer`.

- [ ] **Step 4: Extend `StubRetriever` in `test_deps.py` to support batch search**

Add a `search_filings_batch` method that:

- Accepts `queries: list[str]` and `limit_per_query: int = 5`.
- Sets `last_query` to a representative joined query (e.g. `" | ".join(queries)`), or keeps single-query behavior if tests only cover the one-query case.
- Returns an empty list of `SourcePassage` objects to keep tests simple for now.

- [ ] **Step 5: Add a unit test for compact evidence aliases**

In `test_agent.py` or a new assistant test module, add a test that:

- Creates a concrete `EvidenceRegistry`.
- Mocks `ctx.deps.retriever.search_filings_batch` to return two `SourcePassage` objects with different `chunk_id`s but ensures that `EvidenceRegistry.register` assigns aliases `E1`, `E2`.
- Calls the `search_filings` tool with a list of queries and asserts that:
  - Returned `CompactEvidence` items contain `alias` values like `"E1"`, `"E2"`.
  - There is **no** `chunk_id` field in the serialized (`model_dump()`) tool result.

- [ ] **Step 6: Run focused tests and confirm failures**

Run:

```bash
cd backend && uv run pytest tests/assistant/test_agent.py tests/assistant/test_deps.py -v
```

Confirm failures due to mismatched tool names, missing `GroundedDraft`, and missing `search_filings_batch` on the stub retriever.

---

### Task 2: Extend `DocumentRetriever` protocol and concrete retrievers for batch search

**Files:**
- Modify: `backend/app/assistant/deps.py`
- Modify: `backend/app/retrieval/document_retriever.py`
- Modify: `backend/tests/assistant/test_deps.py`

**Interfaces:**
- Consumes: `RetrievalResult`, `SourcePassage`, `session_scope`, `retrieve_passages`.
- Produces: `DocumentRetriever.search_filings_batch(queries: list[str], *, limit_per_query: int) -> list[SourcePassage]` plus existing methods.

- [ ] **Step 1: Update `DocumentRetriever` protocol**

Add:

```python
def search_filings_batch(
    self,
    queries: list[str],
    *,
    limit_per_query: int = 5,
) -> list[SourcePassage]:
    ...
```

Keep existing `search_filings`, `read_chunk`, and `read_surrounding_chunks` signatures unchanged to preserve orchestrator compatibility.

- [ ] **Step 2: Implement `search_filings_batch` on `SessionDocumentRetriever`**

Follow the brief:

```python
def search_filings_batch(
    self,
    queries: list[str],
    *,
    limit_per_query: int = 5,
) -> list[SourcePassage]:
    seen: dict[UUID, SourcePassage] = {}
    for query in queries:
        result = self.search_filings(query, limit=limit_per_query)
        for passage in result.passages:
            seen.setdefault(passage.chunk_id, passage)
    return list(seen.values())
```

- [ ] **Step 3: Mirror `search_filings_batch` on `SessionPerCallDocumentRetriever`**

Delegate to a `SessionDocumentRetriever` instance inside a `session_scope()` context, mirroring the pattern already used for `search_filings`, `read_chunk`, and `read_surrounding_chunks`.

- [ ] **Step 4: Update any `DEFAULT_RETRIEVAL_LIMIT` uses tied to the agent**

Where the agent tools previously defaulted to `limit=10`, change them to use the budget field `max_hits_per_search` (default 5) when wired through the agent in later tasks. Keep retrieval-layer defaults conservative and let the agent pass explicit limits.

- [ ] **Step 5: Update `StubRetriever` again if needed**

Ensure `StubRetriever.search_filings_batch` returns a `list[SourcePassage]` and that any type hints in tests line up with the new protocol.

- [ ] **Step 6: Run deps tests**

```bash
cd backend && uv run pytest tests/assistant/test_deps.py -v
```

Confirm green for dependency wiring before continuing.

---

### Task 3: Enrich `DocumentAgentDeps` with budget, usage, evidence registry, and search_count

**Files:**
- Modify: `backend/app/assistant/deps.py`
- Modify: `backend/tests/assistant/test_deps.py`

**Interfaces:**
- Consumes: `TurnBudget`, `DEFAULT_TURN_BUDGET`, `TurnUsage`, `EvidenceRegistry`.
- Produces: `DocumentAgentDeps` dataclass with new fields and defaults.

- [ ] **Step 1: Import new types**

In `deps.py`, import:

- `EvidenceRegistry` and `CompactEvidence` (for typing the tool return).
- `TurnBudget`, `DEFAULT_TURN_BUDGET`, and `TurnUsage` from the chat modules.

- [ ] **Step 2: Extend `DocumentAgentDeps` dataclass**

Add fields:

```python
evidence: EvidenceRegistry
budget: TurnBudget = DEFAULT_TURN_BUDGET
usage: TurnUsage
search_count: int = 0
```

Ensure ordering keeps required, non-default fields (`user_id`, `thread_id`, `retriever`, `grounding_validator`, `evidence`, `usage`) before fields with defaults (`budget`, `search_count`).

- [ ] **Step 3: Update `test_document_agent_deps_wires_runtime_services`**

- Construct a `TurnUsage` instance appropriate for a fresh turn (inspecting its constructor or factory).
- Construct an `EvidenceRegistry` instance.
- Update `DocumentAgentDeps` construction to pass `evidence` and `usage` arguments.
- Assert that these fields are present and usable (e.g. `deps.evidence.register([]) == []`, `deps.usage.turn_id` or similar property is set, and `search_count` defaults to 0).

- [ ] **Step 4: Run assistant deps tests**

```bash
cd backend && uv run pytest tests/assistant/test_deps.py -v
```

Confirm that dependency wiring now includes the added fields.

---

### Task 4: Replace agent tools with batched compact `search_filings` and change output type

**Files:**
- Modify: `backend/app/assistant/agent.py`
- Modify: `backend/app/assistant/instructions.md`
- Modify: `backend/tests/assistant/test_agent.py`

**Interfaces:**
- Consumes: `DocumentAgentDeps`, `EvidenceRegistry`, `TurnBudget`, `TurnUsage`, `GroundedDraft`, `CompactEvidence`.
- Produces: New `search_filings` tool signature (`queries: list[str]`) and `document_agent` output type `PromptedOutput(GroundedDraft)`.

- [ ] **Step 1: Change `document_agent` output type**

In `agent.py`:

- Replace import of `GroundedAnswer` with `GroundedDraft`.
- Update `document_agent = Agent(...)` to set `output_type=PromptedOutput(GroundedDraft)`.
- Ensure tests reference `GroundedDraft` when inspecting `result.output`.

- [ ] **Step 2: Remove legacy agent tools**

Delete the `@document_agent.tool` functions:

- `read_chunk`
- `read_surrounding_chunks`

Ensure any imports of `UUID` that become unused are removed to satisfy linters/tests.

- [ ] **Step 3: Implement batched `search_filings` tool using evidence registry and budget**

Replace the existing `search_filings` tool definition with the brief’s implementation:

```python
@document_agent.tool
def search_filings(
    ctx: RunContext[DocumentAgentDeps],
    queries: list[str],
) -> list[CompactEvidence]:
    """Search filings with 1–3 focused queries; returns compact evidence aliases."""
    budget = ctx.deps.budget
    if ctx.deps.search_count >= budget.max_searches:
        return []
    cleaned = [q.strip() for q in queries if q and q.strip()][: budget.max_searches]
    if not cleaned:
        return []
    ctx.deps.search_count += 1
    passages = ctx.deps.retriever.search_filings_batch(
        cleaned,
        limit_per_query=budget.max_hits_per_search,
    )
    compact = ctx.deps.evidence.register(passages)
    ctx.deps.usage.record_passages(len(ctx.deps.evidence.all_passages()))
    return compact
```

Ensure `CompactEvidence` and `EvidenceRegistry` are imported and that `TurnUsage.record_passages` exists and is used correctly.

- [ ] **Step 4: Update instructions to reflect new workflow**

In `instructions.md`:

- Replace the tool section to describe only `search_filings(queries: list[str])` returning compact evidence with aliases like `E1`.
- Remove any references to `read_chunk` and `read_surrounding_chunks`.
- Explain that citations in the answer must use `[n]` markers and `evidence_alias` values like `E1`, not raw `chunk_id`s.
- State that the agent should not invent aliases or ask for / emit UUIDs or full `SourcePassage` objects—only use aliases the tool returned.
- Update the output contract to refer to `GroundedDraft` with `DraftCitation` entries containing `evidence_alias`.

Align test expectations in `test_load_instructions_encodes_grounding_contract` with the new wording.

- [ ] **Step 5: Adjust FunctionModel test flow**

Update the `FunctionModel`-based test to:

- Call `search_filings` once with a `queries` list.
- Ensure the stub retriever’s batch method is invoked, not the old single-query method.
- Assert that the final `result.output` is a `GroundedDraft`.

- [ ] **Step 6: Run assistant tests**

```bash
cd backend && uv run pytest tests/assistant/test_agent.py -v
```

Confirm all assistant tests pass with the new tool surface and output type.

---

### Task 5: Update recording retriever tests if needed

**Files:**
- Modify: `backend/tests/chat/test_recording_retriever.py` (if any expectations tied directly to tool names change)
- Inspect: `backend/app/chat/orchestrator.py` `_RecordingRetriever`

**Interfaces:**
- Consumes: `DocumentRetriever` protocol, `_RecordingRetriever` implementation.
- Produces: Tests that still pass despite the agent no longer exposing `read_chunk` / `read_surrounding_chunks` as tools.

- [ ] **Step 1: Verify `_RecordingRetriever` still compiles**

Check whether `_RecordingRetriever` expects `DocumentRetriever` to expose `read_chunk` and `read_surrounding_chunks` (it should, via the protocol).

Ensure no code paths assume those are registered as agent tools; they should only be callable on the retriever / recorder itself.

- [ ] **Step 2: Adjust tests only if necessary**

If `test_recording_retriever.py` references the agent’s tool registry or tool names, update it to rely on `_RecordingRetriever` methods instead, leaving the tool surface concerns to `test_agent.py`.

If it only exercises `_RecordingRetriever.read_chunk` and `.read_surrounding_chunks`, no changes may be needed.

- [ ] **Step 3: Run recording retriever tests**

```bash
cd backend && uv run pytest tests/chat/test_recording_retriever.py -v
```

Confirm green; keep changes minimal.

---

### Task 6: Full assistant test suite and integration sanity

**Files:**
- Tests only: `backend/tests/assistant/`, `backend/tests/chat/test_recording_retriever.py`

**Interfaces:**
- Consumes: All changes above.
- Produces: Verified TDD cycle and green test suite for affected areas.

- [ ] **Step 1: Run full assistant tests**

```bash
cd backend && uv run pytest tests/assistant/ -v
```

- [ ] **Step 2: Run targeted chat tests**

```bash
cd backend && uv run pytest tests/chat/test_recording_retriever.py -v
```

- [ ] **Step 3: Optionally run focused retrieval tests**

If any retrieval tests exist for batch behavior, run:

```bash
cd backend && uv run pytest tests/retrieval/test_document_retriever.py -v
```

- [ ] **Step 4: Ensure no regressions**

Confirm that no tests elsewhere reference the removed agent tools or `GroundedAnswer` output from the document agent.

---

### Task 7: Commit and reporting

**Files:**
- All modified files from previous tasks
- Report: `.superpowers/sdd/task-4-report.md`

**Interfaces:**
- Produces: Final commit and task report.

- [ ] **Step 1: Stage changes (including instructions.md)**

```bash
cd backend && git status  # sanity check
git add -f app/assistant/instructions.md
cd ..
git add backend/app/assistant backend/app/retrieval/document_retriever.py \
  backend/tests/assistant backend/tests/chat/test_recording_retriever.py \
  docs/superpowers/plans/2026-07-24-task-4-batch-compact-search.md
```

- [ ] **Step 2: Write task report**

Populate `.superpowers/sdd/task-4-report.md` with:

- Summary of behavior changes.
- Tests run (with commands and results).
- Any concerns or follow-ups for Task 5 (orchestrator finalize integration).

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: batch compact search and GroundedDraft agent output"
```

- [ ] **Step 4: Final verification**

Re-run key tests if necessary and ensure working tree is clean.


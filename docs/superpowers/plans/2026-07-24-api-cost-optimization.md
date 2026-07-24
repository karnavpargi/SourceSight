# API Cost Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut chat API cost from ~₹8/answer to under ₹1 while keeping the same grounded answer quality and frontend wire format.

**Architecture:** Replace full passage / UUID tool I/O with a server-owned evidence registry (`E1`…`E8`). The model emits a compact `GroundedDraft`; the backend hydrates `GroundedAnswer`. Cap searches, passages, and output tokens; replace full grounding re-runs with one no-retrieval correction.

**Tech Stack:** FastAPI, PydanticAI (`PromptedOutput`, `ModelSettings.max_tokens`), existing hybrid retriever, pytest.

## Global Constraints

- Target: under ₹1 per answer for the measured Amazon AWS comparison question class.
- Do not reduce visible answer quality; frontend still receives answer + UUID citations + cited passages.
- Gemini context caching is out of scope.
- Do not send chat history (already latest-user-text only).
- Do not commit secrets from `backend/.env`.
- Prefer small focused modules; no new runtime dependencies.
- Follow TDD: failing test → implement → pass → commit per task.

## File map

| File | Responsibility |
|------|----------------|
| `backend/app/chat/turn_budget.py` | Hard caps for searches, passages, output tokens, corrections |
| `backend/app/chat/usage.py` | Cumulative token/call counters for one turn |
| `backend/app/assistant/evidence.py` | Compact evidence models + `EvidenceRegistry` |
| `backend/app/assistant/finalize.py` | `GroundedDraft` → `GroundedAnswer` hydration |
| `backend/app/assistant/outputs.py` | Add `DraftCitation` / `GroundedDraft`; keep `GroundedAnswer` |
| `backend/app/assistant/agent.py` | Single batched search tool; output `GroundedDraft` |
| `backend/app/assistant/deps.py` | Deps carry registry + budget + usage |
| `backend/app/assistant/instructions.md` | Alias citations + batched search workflow |
| `backend/app/chat/generation.py` | Pass `max_tokens` into `ModelSettings` |
| `backend/app/chat/orchestrator.py` | Wire registry/budget; correction without full re-retrieval |
| `backend/app/retrieval/document_retriever.py` | Support batched search + lower default limits |
| `backend/tests/assistant/test_evidence.py` | Registry + compact evidence tests |
| `backend/tests/assistant/test_finalize.py` | Draft finalizer tests |
| `backend/tests/assistant/test_outputs.py` | Draft schema tests |
| `backend/tests/assistant/test_agent.py` | Tool + draft output tests |
| `backend/tests/chat/test_orchestrator.py` | Correction path + usage aggregation |
| `backend/tests/chat/test_generation.py` | max_tokens settings |
| `docs/guides/api-cost-optimization.md` | Already approved design (reference only) |

---

### Task 1: Turn budget and usage recorder

**Files:**
- Create: `backend/app/chat/turn_budget.py`
- Create: `backend/app/chat/usage.py`
- Create: `backend/tests/chat/test_turn_budget.py`
- Create: `backend/tests/chat/test_usage.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `TurnBudget(max_searches=3, max_hits_per_search=5, max_unique_passages=8, max_output_tokens=1500, max_corrections=1)`
  - `TurnUsage` with `add_model_usage(input_tokens, output_tokens)`, `record_embedding()`, `record_passages(n)`, `record_correction()`, and `as_log_fields() -> dict`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/chat/test_turn_budget.py
from app.chat.turn_budget import TurnBudget, DEFAULT_TURN_BUDGET

def test_default_turn_budget_matches_spec() -> None:
    budget = DEFAULT_TURN_BUDGET
    assert budget.max_searches == 3
    assert budget.max_hits_per_search == 5
    assert budget.max_unique_passages == 8
    assert budget.max_output_tokens == 1500
    assert budget.max_corrections == 1
```

```python
# backend/tests/chat/test_usage.py
from app.chat.usage import TurnUsage

def test_turn_usage_aggregates_across_calls() -> None:
    usage = TurnUsage()
    usage.add_model_usage(input_tokens=100, output_tokens=20)
    usage.record_embedding()
    usage.record_passages(5)
    usage.add_model_usage(input_tokens=50, output_tokens=10)
    usage.record_correction()
    fields = usage.as_log_fields()
    assert fields == {
        "model_calls": 2,
        "embedding_calls": 1,
        "input_tokens": 150,
        "output_tokens": 30,
        "passages": 5,
        "corrections": 1,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && uv run pytest tests/chat/test_turn_budget.py tests/chat/test_usage.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `app.chat.turn_budget` / `app.chat.usage`.

- [ ] **Step 3: Implement**

```python
# backend/app/chat/turn_budget.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TurnBudget:
    max_searches: int = 3
    max_hits_per_search: int = 5
    max_unique_passages: int = 8
    max_output_tokens: int = 1500
    max_corrections: int = 1

DEFAULT_TURN_BUDGET = TurnBudget()
```

```python
# backend/app/chat/usage.py
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class TurnUsage:
    model_calls: int = 0
    embedding_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    passages: int = 0
    corrections: int = 0

    def add_model_usage(
        self,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        self.model_calls += 1
        if input_tokens is not None:
            self.input_tokens += input_tokens
        if output_tokens is not None:
            self.output_tokens += output_tokens

    def record_embedding(self) -> None:
        self.embedding_calls += 1

    def record_passages(self, count: int) -> None:
        self.passages = max(self.passages, count)

    def record_correction(self) -> None:
        self.corrections += 1

    def as_log_fields(self) -> dict[str, int]:
        return {
            "model_calls": self.model_calls,
            "embedding_calls": self.embedding_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "passages": self.passages,
            "corrections": self.corrections,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && uv run pytest tests/chat/test_turn_budget.py tests/chat/test_usage.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -f backend/app/chat/turn_budget.py backend/app/chat/usage.py \
  backend/tests/chat/test_turn_budget.py backend/tests/chat/test_usage.py
git commit -m "feat: add chat turn budget and cumulative usage counters"
```

---

### Task 2: Evidence registry and compact evidence

**Files:**
- Create: `backend/app/assistant/evidence.py`
- Create: `backend/tests/assistant/test_evidence.py`

**Interfaces:**
- Consumes: `SourcePassage` from `app.retrieval.types`
- Produces:
  - `CompactEvidence(alias: str, content: str, ticker: str, fiscal_year: int, section: str | None)`
  - `EvidenceRegistry(max_passages: int = 8)`
  - `registry.register(passages: list[SourcePassage]) -> list[CompactEvidence]` (dedupe by chunk_id, stop at max)
  - `registry.resolve(alias: str) -> SourcePassage` (raises `KeyError` for unknown)
  - `registry.all_passages() -> list[SourcePassage]`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/assistant/test_evidence.py
from datetime import date
from uuid import uuid4

from app.assistant.evidence import EvidenceRegistry
from app.retrieval.types import SourcePassage

def _passage(content: str = "AWS income", ticker: str = "AMZN") -> SourcePassage:
    return SourcePassage(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        content=content,
        section="Item 8",
        page=None,
        ticker=ticker,
        company_name="Amazon",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001",
        filing_date=date(2025, 1, 31),
        report_date=None,
        source_url="https://example.com",
        score=1.0,
    )

def test_registry_assigns_aliases_and_dedupes() -> None:
    registry = EvidenceRegistry(max_passages=8)
    p1 = _passage("one")
    p2 = _passage("two")
    first = registry.register([p1, p2, p1])
    assert [e.alias for e in first] == ["E1", "E2"]
    assert first[0].content == "one"
    assert first[0].ticker == "AMZN"
    assert first[0].fiscal_year == 2024
    assert first[0].section == "Item 8"
    assert "chunk_id" not in first[0].model_dump()
    assert registry.resolve("E1").chunk_id == p1.chunk_id

def test_registry_enforces_max_passages() -> None:
    registry = EvidenceRegistry(max_passages=2)
    passages = [_passage(f"c{i}") for i in range(5)]
    compact = registry.register(passages)
    assert len(compact) == 2
    assert len(registry.all_passages()) == 2

def test_registry_rejects_unknown_alias() -> None:
    registry = EvidenceRegistry()
    try:
        registry.resolve("E99")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd backend && uv run pytest tests/assistant/test_evidence.py -v
```

Expected: FAIL with `ModuleNotFoundError: app.assistant.evidence`

- [ ] **Step 3: Implement `evidence.py`**

```python
# backend/app/assistant/evidence.py
from __future__ import annotations
from dataclasses import dataclass, field
from uuid import UUID

from pydantic import BaseModel, Field

from app.retrieval.types import SourcePassage

__all__ = ["CompactEvidence", "EvidenceRegistry"]

class CompactEvidence(BaseModel):
    alias: str = Field(description="Turn-local evidence id, e.g. E1.")
    content: str
    ticker: str
    fiscal_year: int
    section: str | None = None

@dataclass
class EvidenceRegistry:
    max_passages: int = 8
    _by_alias: dict[str, SourcePassage] = field(default_factory=dict)
    _alias_by_chunk: dict[UUID, str] = field(default_factory=dict)

    def register(self, passages: list[SourcePassage]) -> list[CompactEvidence]:
        emitted: list[CompactEvidence] = []
        for passage in passages:
            if passage.chunk_id in self._alias_by_chunk:
                alias = self._alias_by_chunk[passage.chunk_id]
                emitted.append(self._to_compact(alias, self._by_alias[alias]))
                continue
            if len(self._by_alias) >= self.max_passages:
                break
            alias = f"E{len(self._by_alias) + 1}"
            self._by_alias[alias] = passage
            self._alias_by_chunk[passage.chunk_id] = alias
            emitted.append(self._to_compact(alias, passage))
        return emitted

    def resolve(self, alias: str) -> SourcePassage:
        try:
            return self._by_alias[alias]
        except KeyError as exc:
            raise KeyError(f"Unknown evidence alias: {alias}") from exc

    def all_passages(self) -> list[SourcePassage]:
        return list(self._by_alias.values())

    def _to_compact(self, alias: str, passage: SourcePassage) -> CompactEvidence:
        return CompactEvidence(
            alias=alias,
            content=passage.content,
            ticker=passage.ticker,
            fiscal_year=passage.fiscal_year,
            section=passage.section,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && uv run pytest tests/assistant/test_evidence.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -f backend/app/assistant/evidence.py backend/tests/assistant/test_evidence.py
git commit -m "feat: add turn-local evidence registry with compact aliases"
```

---

### Task 3: GroundedDraft schema and server finalizer

**Files:**
- Modify: `backend/app/assistant/outputs.py`
- Create: `backend/app/assistant/finalize.py`
- Modify: `backend/tests/assistant/test_outputs.py`
- Create: `backend/tests/assistant/test_finalize.py`

**Interfaces:**
- Consumes: `EvidenceRegistry`, existing `Citation` / `GroundedAnswer` / `SourcePassage`
- Produces:
  - `DraftCitation(citation_index: int, evidence_alias: str, excerpt: str)`
  - `GroundedDraft(answer: str, citations: list[DraftCitation])` — **no** `cited_passages`, **no** UUID fields
  - `finalize_grounded_draft(draft, registry) -> GroundedAnswer`
  - Raises `ValueError` if any `evidence_alias` is unknown

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/assistant/test_finalize.py
from datetime import date
from uuid import uuid4

from app.assistant.evidence import EvidenceRegistry
from app.assistant.finalize import finalize_grounded_draft
from app.assistant.outputs import DraftCitation, GroundedDraft
from app.retrieval.types import SourcePassage

def test_finalize_maps_aliases_to_uuid_citations_and_passages() -> None:
    passage = SourcePassage(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        content="AWS operating income was $39,834 million.",
        section="Item 8",
        page=None,
        ticker="AMZN",
        company_name="Amazon",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001",
        filing_date=date(2025, 1, 31),
        report_date=None,
        source_url="https://example.com",
        score=1.0,
    )
    registry = EvidenceRegistry()
    registry.register([passage])
    draft = GroundedDraft(
        answer="AWS operating income was $39,834 million [1].",
        citations=[
            DraftCitation(
                citation_index=1,
                evidence_alias="E1",
                excerpt="AWS operating income was $39,834 million.",
            )
        ],
    )
    answer = finalize_grounded_draft(draft, registry)
    assert answer.citations[0].chunk_id == passage.chunk_id
    assert answer.cited_passages == [passage]
    assert answer.answer == draft.answer

def test_finalize_rejects_unknown_alias() -> None:
    draft = GroundedDraft(
        answer="Claim [1].",
        citations=[DraftCitation(citation_index=1, evidence_alias="E9", excerpt="x")],
    )
    try:
        finalize_grounded_draft(draft, EvidenceRegistry())
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "E9" in str(exc)
```

Also extend `test_outputs.py` with:

```python
def test_grounded_draft_has_no_cited_passages_field() -> None:
    fields = GroundedDraft.model_fields
    assert "cited_passages" not in fields
    assert "chunk_id" not in DraftCitation.model_fields
    assert "evidence_alias" in DraftCitation.model_fields
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && uv run pytest tests/assistant/test_finalize.py tests/assistant/test_outputs.py -v
```

Expected: FAIL importing `GroundedDraft` / `finalize_grounded_draft`.

- [ ] **Step 3: Implement**

Add to `outputs.py`:

```python
class DraftCitation(BaseModel):
    citation_index: int = Field(ge=1)
    evidence_alias: str = Field(min_length=2, description="Turn-local alias like E1.")
    excerpt: str = Field(min_length=1)

class GroundedDraft(BaseModel):
    answer: str = Field(min_length=1)
    citations: list[DraftCitation] = Field(default_factory=list)

    @model_validator(mode="after")
    def citation_indices_are_unique(self) -> Self:
        indices = [c.citation_index for c in self.citations]
        if len(indices) != len(set(indices)):
            raise ValueError("citation_index values must be unique within an answer")
        return self
```

```python
# backend/app/assistant/finalize.py
from __future__ import annotations

from app.assistant.evidence import EvidenceRegistry
from app.assistant.outputs import Citation, GroundedAnswer, GroundedDraft

def finalize_grounded_draft(
    draft: GroundedDraft,
    registry: EvidenceRegistry,
) -> GroundedAnswer:
    citations: list[Citation] = []
    cited = []
    seen = set()
    for item in draft.citations:
        try:
            passage = registry.resolve(item.evidence_alias)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        citations.append(
            Citation(
                citation_index=item.citation_index,
                chunk_id=passage.chunk_id,
                excerpt=item.excerpt,
            )
        )
        if passage.chunk_id not in seen:
            cited.append(passage)
            seen.add(passage.chunk_id)
    return GroundedAnswer(
        answer=draft.answer,
        citations=citations,
        cited_passages=cited,
    )
```

Keep existing `GroundedAnswer` unchanged for API/persistence.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && uv run pytest tests/assistant/test_finalize.py tests/assistant/test_outputs.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -f backend/app/assistant/outputs.py backend/app/assistant/finalize.py \
  backend/tests/assistant/test_finalize.py backend/tests/assistant/test_outputs.py
git commit -m "feat: add GroundedDraft and server-side evidence finalizer"
```

---

### Task 4: Batched compact search tool + agent output type

**Files:**
- Modify: `backend/app/assistant/deps.py`
- Modify: `backend/app/assistant/agent.py`
- Modify: `backend/app/assistant/instructions.md`
- Modify: `backend/app/retrieval/document_retriever.py`
- Modify: `backend/tests/assistant/test_agent.py`
- Modify: `backend/tests/assistant/test_deps.py`
- Modify: `backend/tests/chat/test_recording_retriever.py` (if tool surface changes)

**Interfaces:**
- Consumes: `EvidenceRegistry`, `TurnBudget`, existing `retrieve_passages`
- Produces:
  - `DocumentRetriever.search_filings_batch(queries: list[str], *, limit_per_query: int) -> list[SourcePassage]`
  - Agent tool `search_filings(queries: list[str]) -> list[CompactEvidence]` that registers into `ctx.deps.evidence`
  - Remove agent tools `read_chunk` and `read_surrounding_chunks`
  - `document_agent` `output_type=PromptedOutput(GroundedDraft)`
  - `DocumentAgentDeps` fields: `evidence: EvidenceRegistry`, `budget: TurnBudget`, `usage: TurnUsage`, plus existing fields

- [ ] **Step 1: Write failing agent tests**

Update `test_document_agent_registers_retrieval_tools` to expect only `["search_filings"]`.

Update `test_load_instructions_encodes_grounding_contract` to require `GroundedDraft`, `E1`, and `search_filings` with multiple queries; assert `read_chunk` / `read_surrounding_chunks` are absent.

Update the FunctionModel test to call:

```python
ToolCallPart("search_filings", {"queries": ["AMZN AWS operating income 2024"]})
```

and return JSON for `GroundedDraft` (no `cited_passages`).

Add a unit test that the tool returns aliases and never includes `chunk_id` in the dumped tool result.

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && uv run pytest tests/assistant/test_agent.py tests/assistant/test_deps.py -v
```

Expected: FAIL on tool names / output type / deps fields.

- [ ] **Step 3: Implement tool surface**

In `document_retriever.py` add:

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

Mirror on `SessionPerCallDocumentRetriever`.

Change default `DEFAULT_RETRIEVAL_LIMIT` usage from tool default 10 → budget `max_hits_per_search` (5). Keep neighbor_window at 0.

Replace agent tools with:

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

Set `output_type=PromptedOutput(GroundedDraft)`.

Update `instructions.md`:
- Cite with `[n]` markers and `evidence_alias` values like `E1`.
- Use one `search_filings` call with multiple short queries when comparing years/segments.
- Do not invent aliases; only cite returned `E#` values.
- Do not ask for / emit UUIDs or full passage objects.
- Remove `read_chunk` / `read_surrounding_chunks` workflow.

Update stubs in `test_deps.py` to implement `search_filings_batch`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && uv run pytest tests/assistant/ -v
```

Expected: PASS (update any remaining references to old tools/output).

- [ ] **Step 5: Commit**

```bash
git add -f backend/app/assistant backend/app/retrieval/document_retriever.py \
  backend/tests/assistant backend/tests/chat/test_recording_retriever.py
git commit -m "feat: batch compact search and emit GroundedDraft from agent"
```

---

### Task 5: Generation max_tokens + orchestrator correction path

**Files:**
- Modify: `backend/app/chat/generation.py`
- Modify: `backend/app/chat/orchestrator.py`
- Create or modify: `backend/tests/chat/test_generation.py`
- Modify: `backend/tests/chat/test_orchestrator.py`

**Interfaces:**
- Consumes: `TurnBudget`, `TurnUsage`, `finalize_grounded_draft`, `repair_grounded_answer`, `grounding_validator`
- Produces:
  - `build_model_settings(config, *, max_tokens: int) -> ModelSettings` including `max_tokens`
  - `_run_agent` returns `(GroundedAnswer, list[SourcePassage], TurnUsage)` after finalizing draft
  - On `GroundingError` after repair: call `_run_citation_correction` at most once
  - `_run_citation_correction` must **not** call retrieval tools; it prompts with the failed draft + registry compact evidence dump + validator error
  - Second failure → existing `REFUSAL_MESSAGE`
  - `chat.turn_complete` / `chat.agent_complete` logs include `TurnUsage.as_log_fields()`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/chat/test_generation.py
from app.chat.generation import ChatGenerationConfig, build_model_settings

def test_build_model_settings_includes_max_tokens() -> None:
    settings = build_model_settings(ChatGenerationConfig(), max_tokens=1500)
    assert settings["temperature"] == 1.0
    assert settings["max_tokens"] == 1500
```

In `test_orchestrator.py`:
1. Change stubs so agent returns `GroundedDraft` (or mock `_run_agent` still returning `GroundedAnswer` after finalize — prefer testing public stream path with patched `_run_agent_with_retriever`).
2. Add `test_grounding_failure_uses_one_correction_without_extra_retrieval`:
   - First finalize/validate fails with `GroundingError`
   - Correction path called once
   - Retriever search count does not increase during correction
   - Success after correction streams answer
3. Add `test_grounding_correction_failure_refuses` (second failure → refusal).
4. Remove / replace tests that expect a full second `_run_agent` with re-retrieval (`_run_agent_grounding_retry`).

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend && uv run pytest tests/chat/test_generation.py tests/chat/test_orchestrator.py -v
```

Expected: FAIL on `max_tokens` and correction behavior.

- [ ] **Step 3: Implement**

`generation.py`:

```python
def build_model_settings(
    config: ChatGenerationConfig,
    *,
    max_tokens: int,
) -> ModelSettings:
    return {"temperature": config.temperature, "max_tokens": max_tokens}
```

Orchestrator changes (behavioral contract):

1. Construct `EvidenceRegistry(max_passages=budget.max_unique_passages)`, `TurnUsage()`, `TurnBudget` defaults in `_run_agent_with_retriever`.
2. Pass them on `DocumentAgentDeps`.
3. After `document_agent.run`, treat `run.output` as `GroundedDraft`, then `answer = finalize_grounded_draft(run.output, registry)`.
4. Accumulate usage via `_token_usage_fields(run)` into `TurnUsage` (note: PydanticAI usage may already sum tool rounds into one run — still increment `model_calls` once per `document_agent.run`, and count embeddings when search executes if instrumented; minimum bar: log fields from `TurnUsage` after each agent/correction run).
5. Replace `_run_agent_grounding_retry` with:

```python
async def _run_citation_correction(
    *,
    user_text: str,
    failed_draft_answer: str,
    grounding_error: str,
    evidence: EvidenceRegistry,
    chat_model: ResolvedChatModel,
    generation: ChatGenerationConfig,
    usage: TurnUsage,
) -> GroundedDraft:
    # Build a no-tool agent run OR run document_agent with tools disabled and
    # deps.retriever that raises if search is called.
    # Prompt includes: original question, failed answer, error, compact evidence JSON.
    ...
```

Simplest reliable approach for this codebase:
- Keep using `document_agent.run`, but set `deps.budget.max_searches = 0` and give a retriever whose `search_filings_batch` raises `RuntimeError("retrieval disabled during correction")`.
- Preload the same `EvidenceRegistry` instance so aliases still resolve.
- Prompt text tells the model to rewrite citations using only listed aliases.

6. After correction, finalize → repair → validate again. On failure → refuse.
7. Delete the path that re-embeds/re-searches the full question on grounding failure.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && uv run pytest tests/chat/test_generation.py tests/chat/test_orchestrator.py tests/assistant/ tests/grounding/ -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -f backend/app/chat/generation.py backend/app/chat/orchestrator.py \
  backend/app/assistant/deps.py backend/tests/chat
git commit -m "feat: cap output tokens and replace grounding re-run with correction"
```

---

### Task 6: Repair/validator compatibility + full suite

**Files:**
- Modify: `backend/app/grounding/repair.py` only if finalize already fills citations (prefer no change)
- Modify: any tests still constructing agent output with `cited_passages` as model output
- Modify: `backend/app/chat/agent_events.py` / `activity_summary.py` tool labels if tool names changed

**Interfaces:**
- Consumes: finalized `GroundedAnswer` (UUID citations) — validator unchanged
- Produces: green unit suite; activity UI still shows search steps

- [ ] **Step 1: Grep for stale contracts**

```bash
cd backend && rg -n "read_chunk|read_surrounding_chunks|cited_passages|search_filings\(|limit: int = 10|_run_agent_grounding_retry" app tests -g '*.py' -g '*.md'
```

Fix every stale reference that breaks imports or tests.

- [ ] **Step 2: Run full non-integration suite**

```bash
cd backend && uv run pytest -m "not integration" -v
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add -f backend
git commit -m "test: align suite with compact evidence cost controls"
```

---

### Task 7: Manual cost verification (same Amazon question)

**Files:**
- None required (ops check); optionally append measured results to `docs/guides/api-cost-optimization.md`

**Interfaces:**
- Consumes: running backend with Google chat + embeddings
- Produces: log lines proving success criteria

- [ ] **Step 1: Note baseline spend in AI Studio**

Record current Google AI Studio spend before the turn.

- [ ] **Step 2: Run the same measured question**

Use the same prompt:

```text
For Amazon, compare AWS operating income and margin against North America and International from 2021–2025. In which years did AWS appear to fund losses or weaker profitability elsewhere?
```

Against `POST /chat/stream` with `gemini-3.5-flash-lite`.

- [ ] **Step 3: Confirm logs**

From backend logs for that `thread_id`, assert approximately:

- `model_calls` ≤ 4
- `embedding_calls` ≤ 3
- `passages` ≤ 8
- `input_tokens` ≤ 25000
- `output_tokens` ≤ 2000
- `corrections` ≤ 1
- outcome `answered` (or justified refusal)

- [ ] **Step 4: Confirm spend delta under ₹1**

Compare AI Studio spend before/after. Target: delta < ₹1.

- [ ] **Step 5: Commit measurement note if updating the guide**

```bash
git add -f docs/guides/api-cost-optimization.md
git commit -m "docs: record post-optimization cost measurement"
```

Only if you add a short "After" table to the guide.

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| Compact evidence aliases `E1`…`E8` | Task 2 |
| Batched search, ≤3 searches, ≤5 hits, ≤8 passages | Task 4 |
| Model emits draft without full passages/UUIDs | Tasks 3–4 |
| Server hydrates `GroundedAnswer` for UI | Task 3 |
| Output token budget | Task 5 |
| No full grounding re-retrieval; one correction | Task 5 |
| Cumulative usage logging | Tasks 1, 5 |
| Unit tests for registry/finalizer/budget/correction | Tasks 1–6 |
| Manual Amazon cost check | Task 7 |
| Context caching out of scope | (none — intentional) |

## Placeholder / consistency review

- Types: `GroundedDraft` / `DraftCitation` / `CompactEvidence` / `EvidenceRegistry` / `TurnBudget` / `TurnUsage` names are stable across tasks.
- Frontend contract remains `GroundedAnswer` after finalize.
- Old tools `read_chunk` / `read_surrounding_chunks` are removed from the agent (cost); surrounding expansion is out of scope for v1.

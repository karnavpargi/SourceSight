# Multi-Model Cost Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an extract-first, escalate-only Gemini pipeline that answers the ten client-brief question patterns with lower cost while preserving SourceSight's citation contract.

**Architecture:** A no-tool Flash-Lite router emits a validated retrieval plan. The backend performs bounded hybrid retrieval once, optionally expands coverage once, and a no-tool Flash-Lite extractor emits verified facts plus an optional extractive draft. Only synthesis and boundary questions escalate compact facts and excerpts to the user-selected model; server-side alias finalization and deterministic grounding remain authoritative.

**Tech Stack:** Python 3.12+, FastAPI, Pydantic v2, PydanticAI 2.0, SQLAlchemy 2, PostgreSQL/pgvector, structlog, pytest.

## Global Constraints

- Keep the existing frontend `GroundedAnswer` wire shape unchanged.
- Use `CHAT_ROUTER_MODEL` for routing/extraction and the request-selected model for synthesis.
- Router, extractor, and composer agents have no tools and receive no prior chat history.
- Never execute model-generated SQL.
- Standard retrieval: at most 3 total queries, 5 hits/query, 8 unique passages.
- Broad retrieval: at most 5 total queries, 5 hits/query, 15 unique passages.
- Calls per turn: 1 router, 1 extractor, 0–1 synthesis, 0–1 citation correction, 0 full agent retries.
- Output caps: router 300; standard extraction 2,800; broad extraction 3,500; synthesis 2,800; correction 1,200 tokens.
- Unknown model prices produce no estimate; provider billing remains acceptance-test truth.
- No new runtime dependency.
- Do not commit `backend/.env` or credentials.
- Implement with TDD and commit each task independently.

## File map

| File | Responsibility |
|---|---|
| `backend/app/config.py` | Router model and optional exact model-price configuration |
| `backend/app/chat/models_catalog.py` | Resolve the configured Google router model against the live catalog |
| `backend/app/chat/routing.py` | Query plan schemas, fallback plan, plan validation, route decision |
| `backend/app/chat/turn_budget.py` | Standard/broad stage and retrieval budgets |
| `backend/app/chat/usage.py` | Per-stage usage and exact-model cost aggregation |
| `backend/app/retrieval/coverage.py` | Corpus ticker/year coverage and compact prompt summary |
| `backend/app/retrieval/planned.py` | Primary batch retrieval, metadata coverage check, one reserve expansion |
| `backend/app/assistant/facts.py` | Extracted fact schemas and deterministic alias/number validation |
| `backend/app/assistant/router.py` | No-tool query-router agent |
| `backend/app/assistant/extractor.py` | No-tool fact extraction and optional extractive draft |
| `backend/app/assistant/composer.py` | No-tool synthesis/direct-fallback/citation-correction drafts |
| `backend/app/chat/orchestrator.py` | End-to-end stage orchestration and unchanged stream/persistence flow |
| `backend/tests/evaluation/client_brief_questions.py` | Canonical ten-question regression fixtures |

---

### Task 1: Router model configuration and resolution

**Files:**
- Modify: `backend/app/config.py`
- Modify: `backend/app/chat/models_catalog.py`
- Modify: `backend/.env.example`
- Modify: `backend/tests/test_config.py`
- Modify: `backend/tests/chat/test_models_catalog.py`

**Interfaces:**
- Consumes: existing `Settings`, `list_models("google")`, `ResolvedChatModel`.
- Produces:
  - `Settings.chat_router_model: str`
  - `Settings.chat_model_prices: dict[str, tuple[float, float]]`
  - `resolve_router_model() -> ResolvedChatModel | None`
  - `clear_router_model_cache() -> None` for tests/operations.

- [ ] **Step 1: Write failing configuration tests**

Add:

```python
def test_router_model_defaults_to_flash_lite() -> None:
    settings = Settings(
        supabase_url="https://example.supabase.co",
        supabase_anon_key="anon",
        supabase_service_role_key="service",
        database_url="postgresql://postgres:pw@localhost:5432/postgres",
        allowed_origins=["http://localhost:5173"],
        chat_provider="google",
        google_api_key="key",
        _env_file=None,
    )
    assert settings.chat_router_model == "gemini-2.0-flash-lite"


def test_model_prices_parse_exact_model_ids() -> None:
    settings = Settings(
        supabase_url="https://example.supabase.co",
        supabase_anon_key="anon",
        supabase_service_role_key="service",
        database_url="postgresql://postgres:pw@localhost:5432/postgres",
        allowed_origins=["http://localhost:5173"],
        chat_provider="google",
        google_api_key="key",
        chat_model_prices={"gemini-3.5-flash-lite": (0.30, 2.50)},
        _env_file=None,
    )
    assert settings.chat_model_prices["gemini-3.5-flash-lite"] == (0.30, 2.50)
```

Add catalog tests:

```python
def test_resolve_router_model_returns_configured_live_google_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "chat_router_model", "gemini-2.0-flash-lite")
    monkeypatch.setattr(settings, "google_api_key", "key")
    monkeypatch.setattr(
        models_catalog,
        "_cached_google_model_ids",
        lambda: frozenset({"gemini-2.0-flash-lite"}),
    )
    assert resolve_router_model() == ResolvedChatModel(
        provider="google",
        model="gemini-2.0-flash-lite",
    )


def test_resolve_router_model_returns_none_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "chat_router_model", "retired-model")
    monkeypatch.setattr(settings, "google_api_key", "key")
    monkeypatch.setattr(
        models_catalog,
        "_cached_google_model_ids",
        lambda: frozenset({"gemini-3.5-flash-lite"}),
    )
    assert resolve_router_model() is None
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd backend
uv run pytest tests/test_config.py tests/chat/test_models_catalog.py -q
```

Expected: failures for missing settings and resolver.

- [ ] **Step 3: Implement settings and resolver**

Add to `Settings`:

```python
chat_router_model: str = "gemini-2.0-flash-lite"
# Values are paid-tier USD per million (input, output) tokens.
chat_model_prices: dict[str, tuple[float, float]] = {}
```

Validate `chat_router_model.strip()` in `require_provider_keys`.

Add to `models_catalog.py`:

```python
from functools import lru_cache


@lru_cache(maxsize=1)
def _cached_google_model_ids() -> frozenset[str]:
    return frozenset(option.id for option in list_models("google"))


def clear_router_model_cache() -> None:
    _cached_google_model_ids.cache_clear()


def resolve_router_model() -> ResolvedChatModel | None:
    model = settings.chat_router_model.strip()
    if not model or not settings.google_api_key.strip():
        return None
    try:
        available = _cached_google_model_ids()
    except ModelCatalogError:
        return None
    if model not in available:
        return None
    return ResolvedChatModel(provider="google", model=model)
```

Document optional values in `.env.example` without adding prices that may become stale:

```dotenv
CHAT_ROUTER_MODEL=gemini-2.0-flash-lite
# Optional JSON: {"exact-model-id":[input_usd_per_1m,output_usd_per_1m]}
CHAT_MODEL_PRICES={}
```

- [ ] **Step 4: Run tests**

Run:

```bash
cd backend
uv run pytest tests/test_config.py tests/chat/test_models_catalog.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/config.py backend/app/chat/models_catalog.py \
  backend/.env.example backend/tests/test_config.py \
  backend/tests/chat/test_models_catalog.py
git commit -m "feat: configure and resolve cheap router model"
```

---

### Task 2: Query plans and adaptive turn budgets

**Files:**
- Create: `backend/app/chat/routing.py`
- Modify: `backend/app/chat/turn_budget.py`
- Create: `backend/tests/chat/test_routing.py`
- Modify: `backend/tests/chat/test_turn_budget.py`

**Interfaces:**
- Produces:
  - `RouteClass = Literal["extractive", "synthesis", "boundary"]`
  - `QueryPlan`
  - `ValidatedQueryPlan`
  - `fallback_query_plan(question: str) -> QueryPlan`
  - `validate_query_plan(plan, coverage) -> ValidatedQueryPlan`
  - `budget_for_plan(plan: QueryPlan) -> TurnBudget`
  - `STANDARD_TURN_BUDGET`, `BROAD_TURN_BUDGET`.

- [ ] **Step 1: Write failing schema and budget tests**

```python
def test_query_plan_rejects_too_many_queries() -> None:
    with pytest.raises(ValidationError):
        QueryPlan(
            route="synthesis",
            tickers=["AMZN"],
            fiscal_years=[2021, 2022],
            topics=["operating income"],
            primary_queries=["q1", "q2", "q3", "q4"],
            reserve_queries=[],
            requires_synthesis=True,
        )


def test_multi_company_plan_uses_broad_budget() -> None:
    plan = QueryPlan(
        route="synthesis",
        tickers=["AMZN", "MSFT"],
        fiscal_years=[2024],
        topics=["capital expenditure"],
        primary_queries=["AMZN capex", "MSFT capex"],
        reserve_queries=["purchase commitments"],
        requires_synthesis=True,
    )
    assert budget_for_plan(plan) == BROAD_TURN_BUDGET


def test_fallback_plan_uses_one_query_and_synthesis() -> None:
    plan = fallback_query_plan("Compare AWS margins")
    assert plan.route == "synthesis"
    assert plan.primary_queries == ["Compare AWS margins"]
    assert plan.requires_synthesis is True
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
cd backend
uv run pytest tests/chat/test_routing.py tests/chat/test_turn_budget.py -q
```

Expected: import failures for `app.chat.routing` and missing broad budget.

- [ ] **Step 3: Implement routing models**

Create `routing.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

from pydantic import BaseModel, Field, model_validator

if TYPE_CHECKING:
    from app.retrieval.coverage import CorpusCoverage

RouteClass = Literal["extractive", "synthesis", "boundary"]


class QueryPlan(BaseModel):
    route: RouteClass
    tickers: list[str] = Field(default_factory=list, max_length=5)
    fiscal_years: list[int] = Field(default_factory=list, max_length=6)
    topics: list[str] = Field(min_length=1, max_length=8)
    primary_queries: list[str] = Field(min_length=1, max_length=3)
    reserve_queries: list[str] = Field(default_factory=list, max_length=2)
    requires_synthesis: bool

    @model_validator(mode="after")
    def route_matches_synthesis_flag(self) -> "QueryPlan":
        if self.route == "extractive" and self.requires_synthesis:
            raise ValueError("extractive routes cannot require synthesis")
        if self.route == "synthesis" and not self.requires_synthesis:
            raise ValueError("synthesis routes must require synthesis")
        return self


@dataclass(frozen=True)
class ValidatedQueryPlan:
    plan: QueryPlan
    missing_scope: tuple[str, ...]


def fallback_query_plan(question: str) -> QueryPlan:
    return QueryPlan(
        route="synthesis",
        topics=["user question"],
        primary_queries=[question.strip()],
        requires_synthesis=True,
    )


def validate_query_plan(
    plan: QueryPlan,
    coverage: "CorpusCoverage",
) -> ValidatedQueryPlan:
    missing = [
        f"ticker:{ticker}"
        for ticker in plan.tickers
        if ticker not in coverage.tickers
    ]
    missing.extend(
        f"year:{year}"
        for year in plan.fiscal_years
        if year not in coverage.fiscal_years
    )
    return ValidatedQueryPlan(plan=plan, missing_scope=tuple(missing))
```

- [ ] **Step 4: Extend turn budgets**

Replace the budget shape with:

```python
@dataclass(frozen=True)
class TurnBudget:
    max_searches: int
    max_reserve_searches: int
    max_hits_per_search: int
    max_unique_passages: int
    router_output_tokens: int = 300
    extractor_output_tokens: int = 2800
    synthesis_output_tokens: int = 2800
    correction_output_tokens: int = 1200
    max_corrections: int = 1


STANDARD_TURN_BUDGET = TurnBudget(
    max_searches=3,
    max_reserve_searches=1,
    max_hits_per_search=5,
    max_unique_passages=8,
)
BROAD_TURN_BUDGET = TurnBudget(
    max_searches=5,
    max_reserve_searches=2,
    max_hits_per_search=5,
    max_unique_passages=15,
    extractor_output_tokens=3500,
)
DEFAULT_TURN_BUDGET = STANDARD_TURN_BUDGET


def budget_for_plan(plan: QueryPlan) -> TurnBudget:
    return BROAD_TURN_BUDGET if len(set(plan.tickers)) > 1 else STANDARD_TURN_BUDGET
```

Update old tests and call sites that construct `TurnBudget` to pass `max_reserve_searches` and stage-specific token fields.

- [ ] **Step 5: Run tests**

```bash
cd backend
uv run pytest tests/chat/test_routing.py tests/chat/test_turn_budget.py \
  tests/assistant/test_agent.py tests/chat/test_orchestrator.py -q
```

Expected: PASS after call-site updates.

- [ ] **Step 6: Commit**

```bash
git add backend/app/chat/routing.py backend/app/chat/turn_budget.py \
  backend/tests/chat/test_routing.py backend/tests/chat/test_turn_budget.py \
  backend/tests/assistant/test_agent.py backend/tests/chat/test_orchestrator.py
git commit -m "feat: add query plans and adaptive turn budgets"
```

---

### Task 3: Per-stage usage and exact-model cost telemetry

**Files:**
- Modify: `backend/app/chat/usage.py`
- Modify: `backend/tests/chat/test_usage.py`

**Interfaces:**
- Produces:
  - `StageUsage`
  - `TurnUsage.add_model_usage(stage, model, input_tokens, output_tokens)`
  - `TurnUsage.estimated_cost_usd(prices) -> float | None`
  - nested `stages` in `as_log_fields()`.

- [ ] **Step 1: Write failing tests**

```python
def test_turn_usage_tracks_models_by_stage() -> None:
    usage = TurnUsage()
    usage.add_model_usage(
        stage="router",
        model="gemini-2.0-flash-lite",
        input_tokens=100,
        output_tokens=20,
    )
    usage.add_model_usage(
        stage="synthesis",
        model="gemini-3.5-flash-lite",
        input_tokens=500,
        output_tokens=100,
    )
    fields = usage.as_log_fields()
    assert fields["model_calls"] == 2
    assert fields["stages"]["router"]["model"] == "gemini-2.0-flash-lite"
    assert fields["stages"]["synthesis"]["input_tokens"] == 500


def test_estimated_cost_requires_every_exact_model_price() -> None:
    usage = TurnUsage()
    usage.add_model_usage(
        stage="router",
        model="unknown",
        input_tokens=100,
        output_tokens=20,
    )
    assert usage.estimated_cost_usd({}) is None


def test_estimated_cost_uses_per_million_rates() -> None:
    usage = TurnUsage()
    usage.add_model_usage(
        stage="synthesis",
        model="m",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert usage.estimated_cost_usd({"m": (0.30, 2.50)}) == pytest.approx(2.80)
```

- [ ] **Step 2: Run test and verify failure**

```bash
cd backend
uv run pytest tests/chat/test_usage.py -q
```

- [ ] **Step 3: Implement stage aggregation**

```python
@dataclass
class StageUsage:
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class TurnUsage:
    stages: dict[str, StageUsage] = field(default_factory=dict)
    embedding_calls: int = 0
    passages: int = 0
    corrections: int = 0
    route: str | None = None
    budget_profile: str | None = None

    def add_model_usage(
        self,
        *,
        stage: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        current = self.stages.get(stage)
        if current is None:
            current = StageUsage(model=model)
            self.stages[stage] = current
        if current.model != model:
            raise ValueError(f"stage {stage!r} used multiple models")
        current.calls += 1
        current.input_tokens += input_tokens or 0
        current.output_tokens += output_tokens or 0

    def estimated_cost_usd(
        self,
        prices: dict[str, tuple[float, float]],
    ) -> float | None:
        if any(stage.model not in prices for stage in self.stages.values()):
            return None
        return sum(
            (
                stage.input_tokens * prices[stage.model][0]
                + stage.output_tokens * prices[stage.model][1]
            )
            / 1_000_000
            for stage in self.stages.values()
        )
```

`as_log_fields()` must preserve aggregate `model_calls`, `input_tokens`, and `output_tokens`, and add JSON-safe nested stage dictionaries, `route`, and `budget_profile`.

- [ ] **Step 4: Update existing call sites temporarily**

Change current calls to:

```python
usage.add_model_usage(
    stage="synthesis",
    model=chat_model.model,
    **_token_usage_fields(run),
)
```

Correction uses stage `"correction"`.

- [ ] **Step 5: Run focused tests**

```bash
cd backend
uv run pytest tests/chat/test_usage.py tests/chat/test_orchestrator.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/chat/usage.py backend/app/chat/orchestrator.py \
  backend/tests/chat/test_usage.py backend/tests/chat/test_orchestrator.py
git commit -m "feat: record per-stage model usage and cost"
```

---

### Task 4: Corpus coverage and planned retrieval

**Files:**
- Create: `backend/app/retrieval/coverage.py`
- Create: `backend/app/retrieval/planned.py`
- Create: `backend/tests/retrieval/test_coverage.py`
- Create: `backend/tests/retrieval/test_planned.py`

**Interfaces:**
- Produces:
  - `CorpusCoverage(tickers, fiscal_years, ticker_years)`
  - `load_corpus_coverage(session) -> CorpusCoverage`
  - `PlannedRetrieval`
  - `retrieve_for_plan(retriever, validated_plan, budget, usage) -> PlannedRetrieval`.

- [ ] **Step 1: Write failing coverage tests**

```python
def test_coverage_prompt_summary_is_compact_and_sorted() -> None:
    coverage = CorpusCoverage(
        ticker_years={
            "MSFT": frozenset({2023, 2024}),
            "AMZN": frozenset({2022, 2024}),
        }
    )
    assert coverage.prompt_summary() == "AMZN: 2022,2024; MSFT: 2023,2024"


def test_load_corpus_coverage_groups_distinct_rows() -> None:
    session = MagicMock(spec=Session)
    session.execute.return_value.all.return_value = [
        ("AMZN", 2023),
        ("AMZN", 2024),
        ("MSFT", 2024),
    ]
    coverage = load_corpus_coverage(session)
    assert coverage.ticker_years["AMZN"] == frozenset({2023, 2024})
    assert coverage.tickers == frozenset({"AMZN", "MSFT"})
```

- [ ] **Step 2: Write failing planned-retrieval tests**

Use a recording retriever and passages whose ticker/year metadata are controlled:

```python
from dataclasses import dataclass, field
from datetime import date
from uuid import uuid4


def _passage(ticker: str, year: int) -> SourcePassage:
    return SourcePassage(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        content=f"{ticker} filing for {year}",
        section="Item 8",
        ticker=ticker,
        company_name=ticker,
        form_type="10-K",
        fiscal_year=year,
        accession_number=f"{ticker}-{year}",
        filing_date=date(year + 1, 1, 31),
        source_url=f"https://example.com/{ticker}/{year}",
        score=1.0,
    )


def _passages(count: int) -> list[SourcePassage]:
    return [_passage("AMZN", 2024) for _ in range(count)]


def _validated_plan(
    *,
    years: list[int] | None = None,
    primary: list[str],
    reserve: list[str] | None = None,
) -> ValidatedQueryPlan:
    return ValidatedQueryPlan(
        plan=QueryPlan(
            route="extractive",
            tickers=["AMZN"],
            fiscal_years=years or [2024],
            topics=["AWS operating income"],
            primary_queries=primary,
            reserve_queries=reserve or [],
            requires_synthesis=False,
        ),
        missing_scope=(),
    )


@dataclass
class RecordingBatchRetriever:
    batches: list[list[SourcePassage]]
    queries: list[list[str]] = field(default_factory=list)

    def search_filings_batch(
        self,
        queries: list[str],
        *,
        limit_per_query: int = 5,
    ) -> list[SourcePassage]:
        self.queries.append(queries)
        return self.batches[len(self.queries) - 1][: limit_per_query * len(queries)]


def test_retrieve_for_plan_uses_reserve_once_when_year_missing() -> None:
    retriever = RecordingBatchRetriever(
        batches=[
            [_passage("AMZN", 2024)],
            [_passage("AMZN", 2023)],
        ]
    )
    result = retrieve_for_plan(
        retriever,
        _validated_plan(
            years=[2023, 2024],
            primary=["AWS 2024"],
            reserve=["AWS 2023"],
        ),
        STANDARD_TURN_BUDGET,
        TurnUsage(),
    )
    assert retriever.queries == [["AWS 2024"], ["AWS 2023"]]
    assert {(p.ticker, p.fiscal_year) for p in result.passages} == {
        ("AMZN", 2023),
        ("AMZN", 2024),
    }


def test_retrieve_for_plan_enforces_unique_passage_cap() -> None:
    result = retrieve_for_plan(
        RecordingBatchRetriever(batches=[_passages(20)]),
        _validated_plan(primary=["q"]),
        STANDARD_TURN_BUDGET,
        TurnUsage(),
    )
    assert len(result.passages) == 8
```

- [ ] **Step 3: Run tests and confirm failure**

```bash
cd backend
uv run pytest tests/retrieval/test_coverage.py tests/retrieval/test_planned.py -q
```

- [ ] **Step 4: Implement coverage**

```python
class CorpusCoverage(BaseModel):
    ticker_years: dict[str, frozenset[int]]

    @property
    def tickers(self) -> frozenset[str]:
        return frozenset(self.ticker_years)

    @property
    def fiscal_years(self) -> frozenset[int]:
        return frozenset(
            year for years in self.ticker_years.values() for year in years
        )

    def prompt_summary(self) -> str:
        return "; ".join(
            f"{ticker}: {','.join(str(year) for year in sorted(years))}"
            for ticker, years in sorted(self.ticker_years.items())
        )
```

Add to `retrieval/coverage.py`:

```python
def load_corpus_coverage(session: Session) -> CorpusCoverage:
    statement = select(
        SourceDocument.ticker,
        SourceDocument.fiscal_year,
    ).distinct()
    rows = session.execute(statement).all()
    grouped: dict[str, set[int]] = {}
    for ticker, fiscal_year in rows:
        grouped.setdefault(str(ticker), set()).add(int(fiscal_year))
    return CorpusCoverage(
        ticker_years={
            ticker: frozenset(years)
            for ticker, years in grouped.items()
        }
    )
```

Keep `backend/app/database/corpus.py` unchanged; coverage is a retrieval
concern and owns its distinct source-document query.

- [ ] **Step 5: Implement planned retrieval**

```python
@dataclass(frozen=True)
class PlannedRetrieval:
    passages: list[SourcePassage]
    missing_scope: tuple[str, ...]
    expanded: bool


def retrieve_for_plan(
    retriever: DocumentRetriever,
    validated: ValidatedQueryPlan,
    budget: TurnBudget,
    usage: TurnUsage,
) -> PlannedRetrieval:
    plan = validated.plan
    primary = plan.primary_queries[: budget.max_searches]
    passages = retriever.search_filings_batch(
        primary,
        limit_per_query=budget.max_hits_per_search,
    )
    usage.embedding_calls += len(primary)
    passages = _dedupe_and_cap(passages, budget.max_unique_passages)

    missing = _missing_metadata_scope(plan, passages)
    reserve_capacity = budget.max_searches - len(primary)
    reserve = plan.reserve_queries[: min(budget.max_reserve_searches, reserve_capacity)]
    expanded = bool(missing and reserve)
    if expanded:
        extra = retriever.search_filings_batch(
            reserve,
            limit_per_query=budget.max_hits_per_search,
        )
        usage.embedding_calls += len(reserve)
        passages = _dedupe_and_cap(
            [*passages, *extra],
            budget.max_unique_passages,
        )
        missing = _missing_metadata_scope(plan, passages)

    usage.record_passages(len(passages))
    return PlannedRetrieval(
        passages=passages,
        missing_scope=tuple([*validated.missing_scope, *missing]),
        expanded=expanded,
    )
```

`_missing_metadata_scope` checks requested tickers and years independently, not the full ticker/year Cartesian product. `_dedupe_and_cap` preserves retrieval order by using a UUID-keyed dictionary.

- [ ] **Step 6: Run tests**

```bash
cd backend
uv run pytest tests/retrieval/test_coverage.py tests/retrieval/test_planned.py \
  tests/retrieval/test_document_retriever.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/retrieval/coverage.py backend/app/retrieval/planned.py \
  backend/tests/retrieval/test_coverage.py \
  backend/tests/retrieval/test_planned.py
git commit -m "feat: execute validated plans with bounded coverage expansion"
```

---

### Task 5: Extracted facts and deterministic numeric verification

**Files:**
- Create: `backend/app/assistant/facts.py`
- Create: `backend/tests/assistant/test_facts.py`

**Interfaces:**
- Produces:
  - `FactStatus = Literal["supported", "missing", "conflicting"]`
  - `ExtractedFact`
  - `FactExtraction`
  - `ValidatedExtraction`
  - `validate_extraction(extraction, evidence, route) -> ValidatedExtraction`.

- [ ] **Step 1: Write failing model and verification tests**

```python
from datetime import date
from uuid import uuid4


def _registry(content: str) -> EvidenceRegistry:
    registry = EvidenceRegistry()
    registry.register(
        [
            SourcePassage(
                chunk_id=uuid4(),
                document_id=uuid4(),
                chunk_index=0,
                content=content,
                section="Item 8",
                ticker="AMZN",
                company_name="Amazon",
                form_type="10-K",
                fiscal_year=2024,
                accession_number="0001",
                filing_date=date(2025, 1, 31),
                source_url="https://example.com",
                score=1.0,
            )
        ]
    )
    return registry


def _draft(answer: str, alias: str) -> GroundedDraft:
    return GroundedDraft(
        answer=answer,
        citations=[
            DraftCitation(
                citation_index=1,
                evidence_alias=alias,
                excerpt="Revenue was disclosed in the segment table.",
            )
        ],
    )


def test_supported_numeric_fact_requires_known_alias_and_source_value() -> None:
    registry = _registry("Revenue was $39,834 million.")
    extraction = FactExtraction(
        facts=[
            ExtractedFact(
                status="supported",
                ticker="AMZN",
                fiscal_year=2024,
                topic="AWS operating income",
                value="$39,834 million",
                unit="USD millions",
                finding=None,
                evidence_alias="E1",
            )
        ],
        missing_scope=[],
        conflicts=[],
        draft=_draft("AWS operating income was $39,834 million [1].", "E1"),
    )
    validated = validate_extraction(extraction, registry, route="extractive")
    assert len(validated.facts) == 1


def test_numeric_fact_not_present_in_source_is_discarded() -> None:
    registry = _registry("Revenue was $10 million.")
    extraction = FactExtraction(
        facts=[
            ExtractedFact(
                status="supported",
                ticker="AMZN",
                fiscal_year=2024,
                topic="AWS operating income",
                value="$99 million",
                unit="USD millions",
                finding=None,
                evidence_alias="E1",
            )
        ],
        missing_scope=[],
        conflicts=[],
        draft=_draft("AWS operating income was $99 million [1].", "E1"),
    )
    validated = validate_extraction(extraction, registry, route="extractive")
    assert validated.facts == []
    assert "unsupported numeric value" in validated.validation_errors[0]


def test_synthesis_route_rejects_extractive_draft() -> None:
    extraction = FactExtraction(
        facts=[
            ExtractedFact(
                status="supported",
                ticker="AMZN",
                fiscal_year=2024,
                topic="demand",
                value=None,
                unit=None,
                finding="Demand increased.",
                evidence_alias="E1",
            )
        ],
        missing_scope=[],
        conflicts=[],
        draft=_draft("Demand increased [1].", "E1"),
    )
    with pytest.raises(ValueError, match="must omit draft"):
        validate_extraction(
            extraction,
            _registry("Demand increased."),
            route="synthesis",
        )
```

- [ ] **Step 2: Run tests and confirm failure**

```bash
cd backend
uv run pytest tests/assistant/test_facts.py -q
```

- [ ] **Step 3: Implement fact schemas**

```python
FactStatus = Literal["supported", "missing", "conflicting"]


class ExtractedFact(BaseModel):
    status: FactStatus
    ticker: str | None = None
    fiscal_year: int | None = None
    topic: str
    value: str | None = None
    unit: str | None = None
    finding: str | None = None
    evidence_alias: str | None = None

    @model_validator(mode="after")
    def support_has_evidence(self) -> "ExtractedFact":
        if self.status in {"supported", "conflicting"} and not self.evidence_alias:
            raise ValueError("supported/conflicting facts require evidence_alias")
        if self.status == "missing" and self.evidence_alias is not None:
            raise ValueError("missing facts cannot cite evidence")
        return self


class FactExtraction(BaseModel):
    facts: list[ExtractedFact]
    missing_scope: list[str]
    conflicts: list[str]
    draft: GroundedDraft | None = None


@dataclass(frozen=True)
class ValidatedExtraction:
    facts: list[ExtractedFact]
    missing_scope: tuple[str, ...]
    conflicts: tuple[str, ...]
    draft: GroundedDraft | None
    validation_errors: tuple[str, ...]
```

Implement `_numeric_tokens(text)` with a compiled regex, remove commas and currency symbols, normalize parenthesized negatives, and compare every numeric token in `fact.value` against tokens from `registry.resolve(alias).content`. Validate all aliases in facts and draft citations. For `extractive`, require a draft; for `synthesis` and `boundary`, require `draft is None`.

- [ ] **Step 4: Run tests**

```bash
cd backend
uv run pytest tests/assistant/test_facts.py tests/assistant/test_evidence.py \
  tests/assistant/test_finalize.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/assistant/facts.py backend/tests/assistant/test_facts.py
git commit -m "feat: validate extracted facts against server evidence"
```

---

### Task 6: No-tool router, extractor, and composer agents

**Files:**
- Create: `backend/app/assistant/router.py`
- Create: `backend/app/assistant/extractor.py`
- Create: `backend/app/assistant/composer.py`
- Modify: `backend/app/assistant/agent.py`
- Create: `backend/tests/assistant/test_router.py`
- Create: `backend/tests/assistant/test_extractor.py`
- Create: `backend/tests/assistant/test_composer.py`
- Modify: `backend/tests/assistant/test_agent.py`
- Create: `backend/tests/evaluation/client_brief_questions.py`

**Interfaces:**
- Produces:
  - `run_query_router(question, coverage, model, generation, usage, max_tokens) -> QueryPlan`
  - `run_fact_extractor(question, plan, evidence, model, generation, usage, max_tokens) -> FactExtraction`
  - `run_synthesis(question, plan, validated_extraction, evidence, model, generation, usage, max_tokens) -> GroundedDraft`
  - `run_direct_fallback(question, evidence, model, generation, usage, max_tokens) -> GroundedDraft`
  - `run_citation_correction(question, failed_answer, grounding_error, evidence, model, model_name, generation, usage, max_tokens) -> GroundedDraft`.

- [ ] **Step 1: Add canonical ten-question fixtures**

Create a tuple containing the exact ten questions from
`docs/client-brief.md`, plus expected semantic route:

```python
CLIENT_BRIEF_CASES = (
    ClientBriefCase(APPLE_REVENUE_MIX, "extractive"),
    ClientBriefCase(AMAZON_SEGMENTS, "extractive"),
    ClientBriefCase(NVIDIA_DEMAND_DRIVERS, "synthesis"),
    ClientBriefCase(MICROSOFT_AZURE_LANGUAGE, "synthesis"),
    ClientBriefCase(ALPHABET_REVENUE_TRENDS, "extractive"),
    ClientBriefCase(FIVE_COMPANY_RISK_FACTORS, "synthesis"),
    ClientBriefCase(APPLE_NVIDIA_SUPPLIERS, "synthesis"),
    ClientBriefCase(FOUR_COMPANY_CAPEX, "extractive"),
    ClientBriefCase(GEOGRAPHIC_EXPOSURE, "extractive"),
    ClientBriefCase(GENAI_MARGIN_PROOF, "boundary"),
)
```

- [ ] **Step 2: Write failing FunctionModel tests**

For each stage, use PydanticAI's `FunctionModel` and assert:

- router request contains only the question and compact coverage;
- router has zero registered tools;
- extractor request contains aliases but no chunk UUID;
- extractive output contains a draft;
- synthesis output omits a draft;
- composer receives validated facts and compact evidence, not router messages;
- correction has zero tools and uses the correction token cap.

Representative test:

```python
@pytest.mark.anyio
async def test_router_has_no_tools_and_returns_structured_plan() -> None:
    async def model_fn(messages, info):
        assert info.function_tools == []
        assert "AMZN: 2021,2022,2023,2024,2025" in str(messages)
        return ModelResponse(
            parts=[
                TextPart(
                    json.dumps(
                        {
                            "route": "extractive",
                            "tickers": ["AMZN"],
                            "fiscal_years": [2021, 2022, 2023, 2024, 2025],
                            "topics": ["AWS operating income"],
                            "primary_queries": ["AMZN AWS operating income 2021 2025"],
                            "reserve_queries": ["AMZN segment operating income table"],
                            "requires_synthesis": False,
                        }
                    )
                )
            ]
        )

    plan = await run_query_router(
        AMAZON_SEGMENTS,
        _coverage(),
        FunctionModel(model_fn),
        ChatGenerationConfig(),
        TurnUsage(),
        max_tokens=300,
    )
    assert plan.route == "extractive"
```

- [ ] **Step 3: Run tests to confirm failure**

```bash
cd backend
uv run pytest tests/assistant/test_router.py tests/assistant/test_extractor.py \
  tests/assistant/test_composer.py -q
```

- [ ] **Step 4: Implement stage agents**

Move only provider/model construction helpers in `agent.py`; remove the old tool-enabled `document_agent` and `_search_filings_impl` after call sites migrate in Task 7.

Each new module creates one `Agent` with:

```python
Agent(
    build_document_agent_model(settings.chat_provider, "catalog-selected"),
    output_type=PromptedOutput(StageOutputType),
    instructions=STAGE_INSTRUCTIONS,
    retries=0,
)
```

Each runner:

1. serializes only its explicit compact input with `model_dump_json()`;
2. uses `agent.override(model=model)`;
3. passes `build_model_settings(generation, max_tokens=max_tokens)`;
4. calls `usage.add_model_usage` with the concrete stage name, model name, and run token fields; and
5. returns `run.output`.

The extractor prompt explicitly requires an optional draft only for
`extractive`. The composer prompt forbids unsupported facts and requires
aliases from the compact evidence dump. The direct-fallback prompt receives
question + evidence only. Citation correction receives failed answer +
validator error + fixed evidence only.

- [ ] **Step 5: Add route fixture prompt coverage test**

Do not make ten paid model calls. Parameterize the canonical fixtures and
assert each question is included unchanged in the router input generated by
`build_router_prompt`; test expected route values through deterministic
`QueryPlan` fixtures.

- [ ] **Step 6: Run agent tests**

```bash
cd backend
uv run pytest tests/assistant/ -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/app/assistant backend/tests/assistant \
  backend/tests/evaluation/client_brief_questions.py
git commit -m "feat: add no-tool routing extraction and synthesis stages"
```

---

### Task 7: Integrate multi-model stages into the turn orchestrator

**Files:**
- Modify: `backend/app/chat/orchestrator.py`
- Modify: `backend/app/chat/agent_events.py`
- Modify: `backend/tests/chat/test_orchestrator.py`
- Delete obsolete tool-agent expectations from: `backend/tests/chat/test_recording_retriever.py`

**Interfaces:**
- Consumes all prior task interfaces.
- Produces:
  - `_run_routed_turn(user_text, *, user_id, thread_id, chat_model, generation, grounding_validator, retriever, coverage, activity) -> tuple[GroundedAnswer, list[SourcePassage], EvidenceRegistry, TurnUsage]`
  - unchanged `run_chat_turn(client, *, user_id, thread_id, user_text, user_message_data, grounding_validator, chat_model, generation, retriever=None) -> StreamingResponse`.

- [ ] **Step 1: Write failing extractive-route test**

Patch stage runners and assert:

```python
ROUTER_MODEL = ResolvedChatModel(
    provider="google",
    model="gemini-2.0-flash-lite",
)
SYNTHESIS_MODEL = ResolvedChatModel(
    provider="google",
    model="gemini-3.5-flash-lite",
)


def _coverage() -> CorpusCoverage:
    return CorpusCoverage(ticker_years={"AMZN": frozenset({2024})})


def _extractive_plan() -> QueryPlan:
    return QueryPlan(
        route="extractive",
        tickers=["AMZN"],
        fiscal_years=[2024],
        topics=["AWS operating income"],
        primary_queries=["AMZN AWS operating income 2024"],
        reserve_queries=[],
        requires_synthesis=False,
    )


def _extractive_result_with_draft() -> FactExtraction:
    return FactExtraction(
        facts=[
            ExtractedFact(
                status="supported",
                ticker="AMZN",
                fiscal_year=2024,
                topic="AWS operating income",
                value=None,
                unit=None,
                finding="AWS operating income increased.",
                evidence_alias="E1",
            )
        ],
        missing_scope=[],
        conflicts=[],
        draft=GroundedDraft(
            answer="AWS operating income increased [1].",
            citations=[
                DraftCitation(
                    citation_index=1,
                    evidence_alias="E1",
                    excerpt="AWS operating income increased.",
                )
            ],
        ),
    )


@pytest.mark.anyio
async def test_extractive_route_does_not_call_synthesis() -> None:
    router = AsyncMock(return_value=_extractive_plan())
    extractor = AsyncMock(return_value=_extractive_result_with_draft())
    synthesis = AsyncMock()

    with patch(
        "app.chat.orchestrator.resolve_router_model",
        return_value=ROUTER_MODEL,
    ), patch(
        "app.chat.orchestrator.run_query_router",
        new=router,
    ), patch(
        "app.chat.orchestrator.run_fact_extractor",
        new=extractor,
    ), patch(
        "app.chat.orchestrator.run_synthesis",
        new=synthesis,
    ):
        answer, _, _, usage = await _run_routed_turn(
            "Compare AWS operating income",
            user_id=USER_ID,
            thread_id=THREAD_ID,
            chat_model=SYNTHESIS_MODEL,
            generation=ChatGenerationConfig(),
            grounding_validator=grounding_validator,
            retriever=StubRetriever(passages=[_passage()]),
            coverage=_coverage(),
            activity=None,
        )

    synthesis.assert_not_awaited()
    assert answer.answer
    assert usage.route == "extractive"
```

Extend the existing `StubRetriever` with:

```python
def search_filings_batch(
    self,
    queries: list[str],
    *,
    limit_per_query: int = 5,
) -> list[SourcePassage]:
    return self.passages[: limit_per_query * len(queries)]
```

- [ ] **Step 2: Write failing synthesis and boundary tests**

Assert:

- synthesis route calls selected `chat_model` exactly once;
- boundary route can return a grounded limitation;
- router model failure uses `fallback_query_plan`, standard budget, no extractor, and `run_direct_fallback`;
- router model unavailable (`None`) follows the same direct fallback;
- broad multi-company plan uses 15-passage budget;
- retrieval runs via `asyncio.to_thread`, not directly on the event loop;
- no full agent retry occurs;
- correction uses `correction_output_tokens=1200`.

- [ ] **Step 3: Run orchestrator tests and confirm failure**

```bash
cd backend
uv run pytest tests/chat/test_orchestrator.py -q
```

- [ ] **Step 4: Implement routed turn**

Add a focused helper with this flow:

```python
async def _run_routed_turn(
    user_text: str,
    *,
    user_id: UUID,
    thread_id: UUID,
    chat_model: ResolvedChatModel,
    generation: ChatGenerationConfig,
    grounding_validator: GroundingValidator,
    retriever: DocumentRetriever,
    coverage: CorpusCoverage,
    activity: TurnActivityEmitter | None,
) -> tuple[GroundedAnswer, list[SourcePassage], EvidenceRegistry, TurnUsage]:
    usage = TurnUsage()
    router_model = resolve_router_model()

    if router_model is None:
        plan = fallback_query_plan(user_text)
        router_failed = True
    else:
        try:
            plan = await run_query_router(
                user_text,
                coverage,
                build_document_agent_model(
                    router_model.provider,
                    router_model.model,
                ),
                generation,
                usage,
                max_tokens=STANDARD_TURN_BUDGET.router_output_tokens,
            )
            router_failed = False
        except (ModelHTTPError, UnexpectedModelBehavior, ValueError):
            logger.warning("chat.router_fallback", model=router_model.model)
            plan = fallback_query_plan(user_text)
            router_failed = True

    validated = validate_query_plan(plan, coverage)
    budget = budget_for_plan(plan)
    usage.route = plan.route
    usage.budget_profile = "broad" if budget is BROAD_TURN_BUDGET else "standard"

    retrieval = await asyncio.to_thread(
        retrieve_for_plan,
        retriever,
        validated,
        budget,
        usage,
    )
    evidence = EvidenceRegistry(max_passages=budget.max_unique_passages)
    evidence.register(retrieval.passages)

    if router_failed:
        draft = await run_direct_fallback(
            user_text,
            evidence,
            build_document_agent_model(chat_model.provider, chat_model.model),
            chat_model.model,
            generation,
            usage,
            max_tokens=budget.synthesis_output_tokens,
        )
    else:
        extraction = await run_fact_extractor(
            user_text,
            plan,
            evidence,
            build_document_agent_model(router_model.provider, router_model.model),
            router_model.model,
            generation,
            usage,
            max_tokens=budget.extractor_output_tokens,
        )
        checked = validate_extraction(extraction, evidence, route=plan.route)
        missing_scope = (*retrieval.missing_scope, *checked.missing_scope)
        if plan.route == "extractive" and checked.draft is not None:
            draft = checked.draft
        else:
            draft = await run_synthesis(
                user_text,
                plan,
                checked,
                evidence,
                build_document_agent_model(chat_model.provider, chat_model.model),
                chat_model.model,
                generation,
                usage,
                max_tokens=budget.synthesis_output_tokens,
                missing_scope=missing_scope,
            )

    answer = finalize_grounded_draft(draft, evidence)
    return answer, evidence.all_passages(), evidence, usage
```

Align runner parameter order exactly with Task 6's final signatures; do not
duplicate model construction inside stage modules.

- [ ] **Step 5: Load corpus coverage inside the background task**

Add:

```python
def _load_turn_coverage() -> CorpusCoverage:
    with session_scope() as session:
        return load_corpus_coverage(session)
```

Call it with `await asyncio.to_thread(_load_turn_coverage)` inside `_run_agent`
so synchronous database work never blocks SSE emission. Tests inject or patch
coverage.

- [ ] **Step 6: Replace correction implementation**

Delete the nested disabled retriever and old tool-agent correction. Call
`run_citation_correction` from `assistant/composer.py` with fixed evidence and
`budget.correction_output_tokens`. Preserve the existing second validation and
refusal behavior.

- [ ] **Step 7: Include stage telemetry**

At `chat.turn_complete`, add:

```python
estimated_cost_usd=(
    usage.estimated_cost_usd(settings.chat_model_prices)
    if usage is not None
    else None
)
```

Log route, budget profile, retrieval expansion, stage models, and nested stage
usage. Never log prompts, evidence text, API keys, or full user questions.

- [ ] **Step 8: Run focused suites**

```bash
cd backend
uv run pytest tests/chat/test_orchestrator.py tests/chat/test_usage.py \
  tests/assistant/ tests/grounding/ -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add backend/app/chat/orchestrator.py backend/app/chat/agent_events.py \
  backend/tests/chat/test_orchestrator.py \
  backend/tests/chat/test_recording_retriever.py
git commit -m "feat: orchestrate extract-first model routing"
```

---

### Task 8: Full regression, configuration docs, and live evaluation harness

**Files:**
- Create: `backend/tests/evaluation/test_client_brief_routing.py`
- Modify: `backend/tests/grounding/test_adversarial_turns.py`
- Modify: `docs/guides/api-cost-optimization.md`
- Modify: `backend/README.md`

**Interfaces:**
- Produces: a repeatable ten-question offline routing suite and documented live measurement procedure.

- [ ] **Step 1: Add offline ten-question regression**

Parameterize all `CLIENT_BRIEF_CASES` and assert:

- plans validate;
- expected standard/broad profile;
- output call counts never exceed the design;
- no prompt includes previous chat messages;
- extractive fixtures do not invoke synthesis;
- synthesis/boundary fixtures invoke it at most once;
- every successful fixture finalizes aliases; and
- unsupported numeric aliases/refusals retain the grounding contract.

Use FunctionModel responses and stub retrieval only; mark no network or DB.

- [ ] **Step 2: Run the regression and fix contract mismatches**

```bash
cd backend
uv run pytest tests/evaluation/test_client_brief_routing.py \
  tests/grounding/test_adversarial_turns.py -q
```

Expected: PASS.

- [ ] **Step 3: Run static and full backend verification**

```bash
cd backend
uv run ruff check app tests
uv run pytest -m "not integration" -q
```

Expected: Ruff exits 0; at least the existing 285 tests plus new tests pass,
with 4 integration tests deselected unless markers changed intentionally.

- [ ] **Step 4: Document configuration and measurements**

Document:

```dotenv
CHAT_ROUTER_MODEL=gemini-2.0-flash-lite
CHAT_MODEL=gemini-3.5-flash-lite
CHAT_MODEL_PRICES={}
```

In the cost guide, add a "Multi-model routing" section describing stage call
limits, fallback behavior, and that exact prices are operator-configured.
Add a live-results table with columns:

```text
Question | Route | Budget | Router tokens | Extractor tokens |
Synthesis tokens | Correction | Passages | Outcome | Provider cost INR
```

Do not claim the under-₹1 target until provider billing confirms it.

- [ ] **Step 5: Run live evaluation when quota is available**

For each exact question in `docs/client-brief.md`:

1. record provider spend before the request;
2. send one authenticated `/chat/stream` request;
3. capture `chat.turn_complete`;
4. verify valid citations or grounded refusal;
5. record provider spend delta.

Acceptance:

- Amazon question under ₹1 and ≤25,000 input tokens;
- ten-question median under ₹1;
- no normal answer above ₹2 without a logged budget warning;
- no unsupported numerical claim;
- no full agent retry.

If quota is unavailable, document the blocker and do not report projected cost
as measured cost.

- [ ] **Step 6: Commit**

```bash
git add backend/tests/evaluation backend/tests/grounding/test_adversarial_turns.py \
  backend/README.md docs/guides/api-cost-optimization.md
git commit -m "test: cover multi-model routing across client questions"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|---|---|
| Configurable cheap Flash-Lite router | 1 |
| Live-catalog availability and synthesis fallback | 1, 7 |
| Strict no-SQL query plan | 2, 6 |
| Standard/broad retrieval profiles | 2, 4 |
| One metadata-driven reserve expansion before extraction | 4 |
| Typed facts and deterministic numerical validation | 5 |
| Optional extractive draft in the extraction response | 5, 6 |
| Escalate synthesis/boundary only | 6, 7 |
| No-tool stages and no prior chat history | 6, 8 |
| Existing alias finalizer and deterministic grounding | 7 |
| One no-retrieval citation correction | 6, 7 |
| Per-stage usage and exact-price estimate | 3, 7 |
| All ten client-brief questions | 6, 8 |
| Cost and quality acceptance measurements | 8 |

## Self-review

- No runtime dependency is added.
- Query, stage, and budget names are consistent across tasks.
- Retrieval expansion occurs before the single extraction call.
- Extractive drafting is part of extraction, avoiding an undocumented third cheap-model call.
- Router failure performs one selected-model direct fallback rather than two selected-model calls.
- Synchronous DB/retrieval work is explicitly moved off the event loop.
- Live cost is never claimed without provider billing evidence.

## Task 5: Generation max_tokens + orchestrator correction path

### Overview

This task wires per-turn `max_tokens` into model settings, upgrades the orchestrator to work with the new `GroundedDraft` agent output, and replaces the full re-retrieval grounding retry with a single citation-only correction path that reuses existing evidence. It also introduces structured `TurnUsage` aggregation and logging for each turn.

### Implementation

- **Generation settings**
  - Updated `ChatGenerationConfig` consumer in `backend/app/chat/generation.py`:
    - `build_model_settings(config: ChatGenerationConfig, *, max_tokens: int) -> ModelSettings` now returns `{"temperature": config.temperature, "max_tokens": max_tokens}`.
  - All call sites now pass `max_tokens` explicitly from the turn budget:
    - In `backend/app/chat/orchestrator.py`, both initial agent runs and correction runs call `build_model_settings(generation, max_tokens=budget.max_output_tokens)`.
  - Tests:
    - `backend/tests/chat/test_generation.py` now asserts that `build_model_settings` includes the configured temperature and the provided `max_tokens`.

- **Document agent dependencies and retrieval wrapper**
  - `DocumentAgentDeps` was already extended in earlier tasks; orchestrator now provides the full set on each run:
    - In `_run_agent_with_retriever` we construct:
      - `budget: TurnBudget = DEFAULT_TURN_BUDGET`
      - `evidence = EvidenceRegistry(max_passages=budget.max_unique_passages)`
      - `usage = TurnUsage()`
    - These are passed into `DocumentAgentDeps(user_id, thread_id, retriever, grounding_validator, evidence=evidence, usage=usage, budget=budget)`.
  - `_RecordingRetriever` now correctly implements the full `DocumentRetriever` protocol:
    - Added `search_filings_batch(self, queries: list[str], *, limit_per_query: int = 5) -> list[SourcePassage]` which:
      - Emits activity updates with a combined query detail string.
      - Delegates to `inner.search_filings_batch`.
      - Records all returned passages in `seen` and emits an “Analyzing retrieved passages…” step, mirroring `search_filings`.
    - This avoids `AttributeError` when the agent uses the batched search tool while recording retrieval activity.

- **Agent run, finalization, and usage aggregation**
  - `_run_agent` and `_run_agent_with_retriever` now return richer data:
    - `_run_agent(...) -> (GroundedAnswer, list[SourcePassage], EvidenceRegistry, TurnUsage)`
    - `_run_agent_with_retriever(...) -> (GroundedAnswer, list[SourcePassage], EvidenceRegistry, TurnUsage)`
  - `_run_agent` remains responsible for activity “thinking” markers:
    - Starts thinking before delegating to `_run_agent_with_retriever` and ends thinking after the run completes.
  - `_run_agent_with_retriever` is responsible for wiring the agent and aggregating usage:
    - Builds `agent_model` using `build_document_agent_model`.
    - Calls `document_agent.run` with:
      - `deps=DocumentAgentDeps` (including `evidence`, `usage`, `budget` and `_RecordingRetriever`).
      - `model_settings` including the `max_tokens` cap from `TurnBudget`.
      - Optional `event_stream_handler` wired to `agent_event_stream_handler` for live activity updates.
    - After the run:
      - Calls `_token_usage_fields(run)` to extract `input_tokens` / `output_tokens` (falling back to request/response token names when necessary).
      - Calls `usage.add_model_usage(**_token_usage_fields(run))` to increment `model_calls`, `input_tokens`, and `output_tokens`.
      - Logs `chat.agent_complete` with:
        - Provider, model, retrieved passage count, and all `usage.as_log_fields()` (model calls, embedding calls, tokens, passage count, correction count).
    - Finalization with aliases:
      - Treats `run.output` as a `GroundedDraft` and converts it to a `GroundedAnswer` via `finalize_grounded_draft(draft, evidence)`.
      - Returns the finalized `GroundedAnswer`, the recorded `retrieved_passages`, the populated `EvidenceRegistry`, and `TurnUsage`.

- **Repair + validation pipeline**
  - `_finalize_grounded_answer` remains focused on repair and validation:
    - `answer = repair_grounded_answer(answer, retrieved_passages)`
    - `grounding_validator.validate(answer, retrieved_passages)`
    - Returns a `GroundedAnswer` or raises `GroundingError`.
  - `_stream_chat_turn` now:
    - Awaits `_run_agent` and unpacks `(answer, retrieved_passages, evidence, usage)`.
    - Uses `_finalize_grounded_answer` solely for repair + validation, after alias resolution.
    - Carries `usage` through until the final `chat.turn_complete` log, and passes `evidence` into the correction path if needed.

- **Single-pass citation correction (no re-retrieval)**
  - Removed `_run_agent_grounding_retry` and the full re-retrieval loop.
  - Introduced `_run_citation_correction(...) -> GroundedDraft`:
    - Signature:
      - `user_text: str`
      - `failed_draft_answer: str`
      - `grounding_error: str`
      - `evidence: EvidenceRegistry`
      - `chat_model: ResolvedChatModel`
      - `generation: ChatGenerationConfig`
      - `usage: TurnUsage`
      - `user_id: UUID`
      - `thread_id: UUID`
    - Prompt construction:
      - Builds a compact JSON representation of aliased evidence from `EvidenceRegistry`:
        - Iterates `evidence._by_alias` (alias → `SourcePassage`).
        - Serializes `{"alias", "content", "ticker", "fiscal_year", "section"}` per alias.
      - Prompt includes:
        - Original user question.
        - The validator error message.
        - The previous answer text (`failed_draft_answer`).
        - The compact evidence JSON keyed by alias.
        - Clear instructions:
          - Fix citation placement for all $ / % / numeric claims.
          - Do not call tools or retrieve new documents.
          - Use only the provided evidence aliases.
    - Retrieval is explicitly disabled:
      - Uses a `_DisabledRetriever` implementation whose `search_filings`, `search_filings_batch`, `read_chunk`, and `read_surrounding_chunks` all raise `RuntimeError("retrieval disabled during correction")` if invoked.
      - Uses a `_NoOpValidator` as the `GroundingValidator` for this run (actual validation occurs afterward in `_finalize_grounded_answer`).
      - Sets a `TurnBudget` clone with `max_searches=0` while preserving other limits from `DEFAULT_TURN_BUDGET`.
    - Runs `document_agent.run` with:
      - The correction prompt.
      - `deps=DocumentAgentDeps(..., retriever=_DisabledRetriever(), grounding_validator=_NoOpValidator(), evidence=evidence, usage=usage, budget=budget)`.
      - `model_settings` with the same `max_output_tokens` cap.
    - After correction:
      - Calls `usage.record_correction()` and `usage.add_model_usage(**_token_usage_fields(run))`.
      - Logs another `chat.agent_complete` with `correction=True` and aggregated usage fields.
      - Returns the corrected `GroundedDraft`.
  - Grounding error handling in `_stream_chat_turn`:
    - On `GroundingError` from `_finalize_grounded_answer`:
      - If there are citations, retrieved passages, and both `evidence` and `usage` are non-`None`:
        - Logs `chat.grounding_correction` with reason, provider, model, retrieved passage count, and citation count.
        - Calls `_run_citation_correction(...)` once.
        - Finalizes the corrected draft with `finalize_grounded_draft(draft, evidence)`.
        - Refreshes `retrieved_passages` from `evidence.all_passages()`.
        - Re-runs `_finalize_grounded_answer` on the corrected answer.
        - If the second validation passes, the turn proceeds normally.
        - If the second validation raises `GroundingError`, no further attempts are made; the code falls through to the refusal path.
      - If there are no citations or no evidence, the orchestrator does not attempt correction and immediately refuses.
    - The refusal path (after zero or one failed correction) remains:
      - Writes a validation-failed activity step.
      - Appends a refusal message using the existing `REFUSAL_MESSAGE`.
      - Does not save or attach citations.
      - Logs `chat.turn_complete` with outcome `"grounding_refusal"` and the aggregated `TurnUsage` fields.

- **TurnUsage aggregation into logs**
  - `_log_turn_complete` signature extended with `usage: TurnUsage | None = None`.
    - When `usage` is provided, `usage.as_log_fields()` is injected into the structured log event:
      - `model_calls`, `embedding_calls`, `input_tokens`, `output_tokens`, `passages`, `corrections`.
  - Call sites:
    - Success (`outcome="answered"`): passes `usage` and `citation_count=len(answer.citations)`.
    - Grounding refusal (`outcome="grounding_refusal"`): passes `usage`.
    - Model-unavailable and generic failure paths pass `usage` when available; otherwise, they log without usage fields.
  - `chat.agent_complete` logs now also include the full `TurnUsage` fields instead of raw run token counts, ensuring that downstream analytics see a coherent per-turn view even when multiple agent runs (initial + correction) occurred.

### Tests

- **Updated tests**
  - `backend/tests/chat/test_generation.py`
    - Added `test_build_model_settings_includes_max_tokens`.
    - Adjusted existing `test_build_model_settings_uses_temperature` to use the new keyword parameter and still assert temperature is respected.
  - `backend/tests/chat/test_orchestrator.py`
    - All stubs for `_run_agent` now return `(GroundedAnswer, [SourcePassage], EvidenceRegistry, TurnUsage)` to match the new interface.
    - `test_run_chat_turn_refuses_on_grounding_failure`:
      - Uses `RecordingValidator(should_fail=True)` plus a patched `_run_citation_correction` that returns a `GroundedDraft` with the same answer text; asserts that the client sees a refusal and no citations or saves.
    - `test_run_chat_turn_refuses_ungrounded_model_answer`:
      - Uses `_fake_run_agent_ungrounded` to simulate an answer with no citations and verifies the refusal path without touching correction.
    - New tests replacing the old full re-run assertions:
      - `test_grounding_failure_uses_one_correction_without_extra_retrieval`:
        - Patches `_run_agent` to return a partially grounded answer that should fail validation once.
        - Patches `_finalize_grounded_answer` to raise `GroundingError` on the first call and succeed on the second.
        - Patches `_run_citation_correction` to return a corrected `GroundedDraft`.
        - Asserts:
          - The final streamed answer matches the corrected text.
          - `_finalize_grounded_answer` is called exactly twice (initial + correction).
          - `_run_citation_correction` is called exactly once.
          - `_run_agent` is only called once (no second full agent run).
      - `test_grounding_correction_failure_refuses`:
        - Patches `_run_agent` to return a bad answer.
        - Patches `_finalize_grounded_answer` to raise `GroundingError` on both calls.
        - Patches `_run_citation_correction` to return a `GroundedDraft`.
        - Asserts:
          - Final SSE body is `REFUSAL_MESSAGE`.
          - `_run_agent` is called once, `_run_citation_correction` once.
          - No citations are attached and message data is not updated.
      - `test_run_chat_turn_does_not_retry_without_citations`:
        - Patches `_run_agent` to return an answer with no citations.
        - Patches `_finalize_grounded_answer` to always raise `GroundingError`.
        - Patches `_run_citation_correction` with an `AsyncMock`.
        - Asserts:
          - Final SSE body is `REFUSAL_MESSAGE`.
          - `_run_citation_correction` is never awaited.
          - No citation attachments occur.
    - Existing tests (`test_run_chat_turn_emits_progress_before_answer`, `test_run_chat_turn_titles_thread_from_first_message`, `test_run_chat_turn_streams_model_unavailable_message`) were updated to align with the new `_run_agent` return shape but otherwise continue to assert the same streaming behavior.
  - `backend/tests/grounding/test_adversarial_turns.py`
    - `StubRetriever` remains unchanged and is now compatible with the extended `_RecordingRetriever`.
    - `_fake_run_agent_factory` was updated to return `(GroundedAnswer, list[SourcePassage], EvidenceRegistry, TurnUsage)`; all adversarial and happy-path tests continue to patch `_run_agent` and verify:
      - Refusals for out-of-corpus companies and investment advice.
      - Refusal of uncited answers.
      - Repair of partial answers with markers but missing citation records.
      - Normal streaming when answers are well grounded.

- **Test command run**
  - From `backend/` inside the `api-cost-optimization` worktree:

    ```bash
    uv run pytest tests/chat/test_generation.py tests/chat/test_orchestrator.py tests/assistant/ tests/grounding/ -v
    ```

  - Result: **53 tests passed**, 0 failed (1 warning from `google.genai` deprecation).

### Notes and Considerations

- The correction prompt uses the existing `EvidenceRegistry` and its alias mapping, so any aliases (`E1`, `E2`, …) established during the initial tool calls are preserved and surfaced back to the model without re-querying the corpus.
- Retrieval is fully disabled during the correction run:
  - `TurnBudget.max_searches` is set to zero.
  - The injected retriever raises on any retrieval call.
  - This ensures correction runs cannot introduce new evidence or silently expand the evidence set.
- `TurnUsage` now reflects both the initial agent run and any correction runs within a turn; downstream logs can distinguish correction-heavy turns (via the `corrections` field) and total token usage regardless of how many model calls occurred.


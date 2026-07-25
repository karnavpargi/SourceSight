# Chat usage footer

**Date:** 2026-07-25
**Status:** Approved

## Goal

Show token usage and estimated model cost beneath each completed assistant
answer. The footer must work for both a newly streamed answer and a historical
message loaded from the database.

## Display

The footer is a single muted line below the source citations:

```text
9.7k input · 2.4k output · ~$0.0088
```

Token counts use compact decimal formatting. Cost uses enough decimal places to
avoid displaying a small non-zero charge as `$0.00`.

If exact prices are unavailable for any model used by the turn,
`estimated_cost_usd` is `null`; the UI then shows only input and output tokens.
The footer is hidden when no usage part exists, which keeps older messages
unchanged.

## Data contract

Add one custom AI SDK message part:

```text
type: data-usage
data:
  input_tokens: integer
  output_tokens: integer
  estimated_cost_usd: number | null
```

The backend derives this payload from the existing `TurnUsage` object and
`CHAT_MODEL_PRICES`. It emits the part once in the live SSE stream and stores
the same part in the assistant message's `message_data`.

The usage payload does not expose prompts, evidence, API keys, stage details,
or pricing configuration.

## Backend changes

- Add typed usage data and `data-usage` parsing alongside the existing custom
  activity, citation, and source-passage parts.
- Extend assistant message serialization to accept optional usage data.
- Persist usage with the completed assistant answer.
- Emit the usage part during the completed live stream so the footer appears
  without a page reload.
- Keep logging unchanged; `chat.turn_complete` remains the server-side detailed
  usage record.

## Frontend changes

- Extend `SourceSightUIMessage` with the `data-usage` part.
- Add a helper that returns the final usage payload from a message.
- Render a small footer under `SourceCitations` for assistant messages.
- Keep route, model, stage counts, and per-stage token details out of this
  compact version.

## Error handling

Usage display is informational. Missing or malformed usage data must not hide
the answer, citations, or activity history. Existing messages without usage
continue to render normally.

## Tests

- Backend serialization round-trip for `data-usage`.
- Backend persistence includes usage for completed assistant answers.
- Streaming emits exactly one usage part with token totals and optional cost.
- Frontend helper reads usage and returns `undefined` for older messages.
- Footer formatting covers thousands, small non-zero costs, and null cost.
- Message rendering includes the footer only when usage exists.

## Out of scope

- Per-stage usage panels.
- Currency conversion to INR.
- Provider billing reconciliation.
- Usage aggregation across threads or users.

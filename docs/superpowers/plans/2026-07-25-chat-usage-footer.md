# Chat Usage Footer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show input tokens, output tokens, and optional estimated USD cost beneath each completed assistant answer, both live and after a history reload.

**Architecture:** Add a typed `data-usage` AI SDK part to the same backend message pipeline used by activity, citation, and source-passage parts. Build the payload once from the final `TurnUsage`, emit it in the SSE response, persist it in `message_data`, and render a compact frontend footer.

**Tech Stack:** Python 3.12+, FastAPI, Pydantic v2, pytest, React 19, TypeScript 6, AI SDK UI messages, Node's built-in test runner, Tailwind CSS.

## Global Constraints

- The wire payload contains only `input_tokens`, `output_tokens`, and `estimated_cost_usd`.
- Tokens are non-negative integers; cost is a non-negative number or `null`.
- If pricing is incomplete, show tokens and omit cost.
- Older messages without `data-usage` render unchanged.
- Persisted and live messages use the same payload.
- Emit exactly one usage part per successful answered turn.
- Do not expose prompts, evidence, API keys, per-stage details, or pricing configuration.
- Do not add runtime or test dependencies.
- Keep the existing `GroundedAnswer` shape unchanged.
- Keep current unrelated working-tree changes out of these task commits.

---

### Task 1: Carry usage through the backend message wire

**Files:**
- Modify: `backend/app/chat/messages.py`
- Modify: `backend/app/chat/persistence.py`
- Modify: `backend/app/chat/streaming.py`
- Modify: `backend/app/chat/orchestrator.py`
- Modify: `backend/tests/chat/test_messages.py`
- Modify: `backend/tests/chat/test_persistence.py`
- Modify: `backend/tests/chat/test_streaming.py`
- Modify: `backend/tests/chat/test_orchestrator.py`

**Interfaces:**
- Consumes: `TurnUsage.input_tokens`, `TurnUsage.output_tokens`, `TurnUsage.estimated_cost_usd(prices)`.
- Produces: `UsageData(input_tokens: int, output_tokens: int, estimated_cost_usd: float | None)`.
- Produces: `UsagePart(type="data-usage", data=UsageData(...))`.
- Extends: `grounded_answer_to_ui_message(..., usage: UsageData | None = None)`.
- Extends: `assistant_answer_to_wire(..., usage: UsageData | None = None)`.
- Extends: `stream_grounded_answer_events(..., usage: UsageData | None = None)`.
- Extends: `_persist_assistant_answer(..., usage: UsageData | None = None)`.

- [ ] **Step 1: Add failing message parsing and serialization tests**

In `backend/tests/chat/test_messages.py`, import `UsageData`, then add:

```python
def test_usage_part_round_trips_through_ui_message() -> None:
    raw = {
        "id": "assistant-1",
        "role": "assistant",
        "parts": [
            {
                "type": "data-usage",
                "data": {
                    "input_tokens": 9729,
                    "output_tokens": 2372,
                    "estimated_cost_usd": 0.008849,
                },
            }
        ],
    }

    parsed = parse_ui_message(raw)

    assert ui_message_to_wire(parsed) == raw
```

Add boundary validation:

```python
@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
def test_usage_data_rejects_negative_tokens(field: str) -> None:
    payload = {
        "input_tokens": 1,
        "output_tokens": 1,
        "estimated_cost_usd": None,
    }
    payload[field] = -1

    with pytest.raises(ValidationError):
        UsageData.model_validate(payload)
```

- [ ] **Step 2: Run the message tests and confirm failure**

Run:

```bash
cd backend
uv run pytest tests/chat/test_messages.py -q
```

Expected: collection or assertion failure because `UsageData` and `data-usage` do not exist.

- [ ] **Step 3: Add the typed backend usage part**

In `backend/app/chat/messages.py`, add:

```python
class UsageData(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class UsagePart(BaseModel):
    type: Literal["data-usage"] = "data-usage"
    data: UsageData
    id: str | None = None
```

Add `UsagePart` to `ChatUIPart`. Extend `_parse_part`:

```python
if part_type == "data-usage":
    return UsagePart.model_validate(data)
```

Extend `grounded_answer_to_ui_message`:

```python
def grounded_answer_to_ui_message(
    answer: GroundedAnswer,
    *,
    message_id: str,
    activity_steps: list[TurnActivityData] | None = None,
    usage: UsageData | None = None,
) -> ChatUIMessage:
```

Append usage after citations and source passages:

```python
if usage is not None:
    parts.append(UsagePart(data=usage))
```

- [ ] **Step 4: Run the message tests**

Run:

```bash
cd backend
uv run pytest tests/chat/test_messages.py -q
```

Expected: PASS.

- [ ] **Step 5: Add failing persistence and streaming tests**

In `backend/tests/chat/test_persistence.py`, extend the existing
`assistant_answer_to_wire` test with:

```python
usage = UsageData(
    input_tokens=9729,
    output_tokens=2372,
    estimated_cost_usd=0.008849,
)
wire = assistant_answer_to_wire(
    answer,
    message_id="assistant-1",
    usage=usage,
)
assert wire["parts"][-1] == {
    "type": "data-usage",
    "data": {
        "input_tokens": 9729,
        "output_tokens": 2372,
        "estimated_cost_usd": 0.008849,
    },
}
```

In `backend/tests/chat/test_streaming.py`, add:

```python
@pytest.mark.anyio
async def test_grounded_stream_emits_one_usage_part() -> None:
    usage = UsageData(
        input_tokens=9729,
        output_tokens=2372,
        estimated_cost_usd=0.008849,
    )

    events = [
        event
        async for event in stream_grounded_answer_events(
            _answer(),
            message_id="assistant-1",
            usage=usage,
        )
    ]
    payloads = [
        json.loads(event.removeprefix("data: ").strip())
        for event in events
        if event.startswith("data: {")
    ]
    usage_parts = [payload for payload in payloads if payload["type"] == "data-usage"]

    assert usage_parts == [
        {
            "type": "data-usage",
            "data": {
                "input_tokens": 9729,
                "output_tokens": 2372,
                "estimated_cost_usd": 0.008849,
            },
        }
    ]
```

Use the test file's existing grounded-answer fixture instead of creating a
second fixture if its name differs from `_answer`.

- [ ] **Step 6: Run persistence and streaming tests and confirm failure**

Run:

```bash
cd backend
uv run pytest tests/chat/test_persistence.py tests/chat/test_streaming.py -q
```

Expected: FAIL because the serializers do not accept `usage`.

- [ ] **Step 7: Thread optional usage through persistence and streaming**

In `backend/app/chat/persistence.py`:

```python
def assistant_answer_to_wire(
    answer: GroundedAnswer,
    *,
    message_id: str,
    activity_steps: list[TurnActivityData] | None = None,
    usage: UsageData | None = None,
) -> dict:
    ui_message = grounded_answer_to_ui_message(
        answer,
        message_id=message_id,
        activity_steps=activity_steps,
        usage=usage,
    )
    return ui_message_to_wire(ui_message)
```

In `backend/app/chat/streaming.py`:

```python
async def stream_grounded_answer_events(
    answer: GroundedAnswer,
    *,
    message_id: str | None = None,
    include_start: bool = True,
    usage: UsageData | None = None,
) -> AsyncIterator[str]:
```

Pass `usage=usage` to `grounded_answer_to_ui_message`.

- [ ] **Step 8: Run persistence and streaming tests**

Run:

```bash
cd backend
uv run pytest tests/chat/test_persistence.py tests/chat/test_streaming.py -q
```

Expected: PASS.

- [ ] **Step 9: Add failing orchestrator coverage**

In the successful answered-turn test in
`backend/tests/chat/test_orchestrator.py`, configure a `TurnUsage` with known
stage totals and patch prices:

```python
usage = TurnUsage()
usage.record_model(
    "router",
    model="gemini-flash-lite-latest",
    input_tokens=596,
    output_tokens=214,
)
usage.record_model(
    "synthesis",
    model="gemini-3.5-flash-lite",
    input_tokens=4902,
    output_tokens=920,
)
monkeypatch.setattr(
    settings,
    "chat_model_prices",
    {
        "gemini-flash-lite-latest": (0.30, 2.50),
        "gemini-3.5-flash-lite": (0.30, 2.50),
    },
)
```

After collecting the stream and inspecting
`chat_store.update_message_data.await_args`, assert:

```python
usage_parts = [
    part
    for part in persisted_message["parts"]
    if part["type"] == "data-usage"
]
assert len(usage_parts) == 1
assert usage_parts[0]["data"]["input_tokens"] == 5498
assert usage_parts[0]["data"]["output_tokens"] == 1134
assert usage_parts[0]["data"]["estimated_cost_usd"] == pytest.approx(0.0044844)
assert body.count('"type":"data-usage"') == 1
```

Add a second case with `settings.chat_model_prices = {}` and assert the
persisted usage payload has `estimated_cost_usd is None`.

- [ ] **Step 10: Run the orchestrator tests and confirm failure**

Run:

```bash
cd backend
uv run pytest tests/chat/test_orchestrator.py -q
```

Expected: FAIL because `_stream_chat_turn` does not create or pass usage data.

- [ ] **Step 11: Build the final usage payload once in the orchestrator**

In `backend/app/chat/orchestrator.py`, after grounding/correction finishes and
before `_persist_assistant_answer`, build:

```python
usage_data = UsageData(
    input_tokens=usage.input_tokens,
    output_tokens=usage.output_tokens,
    estimated_cost_usd=usage.estimated_cost_usd(settings.chat_model_prices),
)
```

Pass `usage=usage_data` to `_persist_assistant_answer` and
`stream_grounded_answer_events`.

Extend `_persist_assistant_answer`:

```python
async def _persist_assistant_answer(
    client: AsyncClient,
    *,
    user_id: UUID,
    thread_id: UUID,
    answer: GroundedAnswer,
    activity_log: list[TurnActivityData] | None = None,
    usage: UsageData | None = None,
) -> None:
```

Pass `usage=usage` to `assistant_answer_to_wire`.

- [ ] **Step 12: Run all affected backend tests**

Run:

```bash
cd backend
uv run pytest tests/chat/test_messages.py tests/chat/test_persistence.py \
  tests/chat/test_streaming.py tests/chat/test_orchestrator.py -q
uv run ruff check app tests
```

Expected: all tests PASS; Ruff exits 0.

- [ ] **Step 13: Commit the backend wire change**

```bash
git add backend/app/chat/messages.py backend/app/chat/persistence.py \
  backend/app/chat/streaming.py backend/app/chat/orchestrator.py \
  backend/tests/chat/test_messages.py backend/tests/chat/test_persistence.py \
  backend/tests/chat/test_streaming.py backend/tests/chat/test_orchestrator.py
git commit -m "feat: stream and persist chat usage"
```

---

### Task 2: Render the compact usage footer

**Files:**
- Modify: `frontend/src/lib/chat.ts`
- Create: `frontend/src/lib/chat.test.ts`
- Create: `frontend/src/components/chat/UsageFooter.tsx`
- Modify: `frontend/src/components/chat/MessageList.tsx`
- Modify: `frontend/package.json`
- Modify: `docs/guides/api-cost-optimization.md`

**Interfaces:**
- Consumes: backend `data-usage` with `input_tokens`, `output_tokens`, and `estimated_cost_usd`.
- Produces: `TurnUsageData`.
- Produces: `messageUsage(message: UIMessage): TurnUsageData | undefined`.
- Produces: `formatUsageFooter(usage: TurnUsageData): string`.
- Produces: `UsageFooter({ usage }: { usage: TurnUsageData }): JSX.Element`.

- [ ] **Step 1: Add the native Node test command**

In `frontend/package.json`, add:

```json
"test": "node --test src/lib/*.test.ts"
```

No package installation or lockfile change is needed; Node 24 runs erasable
TypeScript directly.

- [ ] **Step 2: Write failing helper and formatter tests**

Create `frontend/src/lib/chat.test.ts`:

```typescript
import assert from 'node:assert/strict'
import test from 'node:test'

import {
  formatUsageFooter,
  messageUsage,
  type SourceSightUIMessage,
} from './chat.ts'

test('messageUsage returns the final valid usage part', () => {
  const message = {
    id: 'assistant-1',
    role: 'assistant',
    parts: [
      { type: 'text', text: 'Answer' },
      {
        type: 'data-usage',
        data: {
          input_tokens: 9729,
          output_tokens: 2372,
          estimated_cost_usd: 0.008849,
        },
      },
    ],
  } as SourceSightUIMessage

  assert.deepEqual(messageUsage(message), {
    input_tokens: 9729,
    output_tokens: 2372,
    estimated_cost_usd: 0.008849,
  })
})

test('messageUsage ignores missing and malformed usage parts', () => {
  const oldMessage = {
    id: 'assistant-old',
    role: 'assistant',
    parts: [{ type: 'text', text: 'Old answer' }],
  } as SourceSightUIMessage

  assert.equal(messageUsage(oldMessage), undefined)
})

test('formatUsageFooter formats tokens and a small non-zero cost', () => {
  assert.equal(
    formatUsageFooter({
      input_tokens: 9729,
      output_tokens: 2372,
      estimated_cost_usd: 0.008849,
    }),
    '9.7k input · 2.4k output · ~$0.0088',
  )
})

test('formatUsageFooter omits unavailable cost', () => {
  assert.equal(
    formatUsageFooter({
      input_tokens: 9729,
      output_tokens: 2372,
      estimated_cost_usd: null,
    }),
    '9.7k input · 2.4k output',
  )
})
```

- [ ] **Step 3: Run the frontend tests and confirm failure**

Run:

```bash
cd frontend
npm test
```

Expected: FAIL because the usage functions and type do not exist.

- [ ] **Step 4: Add the usage type, guard, accessor, and formatter**

In `frontend/src/lib/chat.ts`, add:

```typescript
export interface TurnUsageData {
  input_tokens: number
  output_tokens: number
  estimated_cost_usd: number | null
}
```

Add `usage: TurnUsageData` to the `SourceSightUIMessage` data map.

Add:

```typescript
export function messageUsage(message: UIMessage): TurnUsageData | undefined {
  let latest: TurnUsageData | undefined

  for (const part of message.parts) {
    if (part.type !== 'data-usage' || !isTurnUsageData(part.data)) {
      continue
    }
    latest = part.data
  }

  return latest
}

export function formatUsageFooter(usage: TurnUsageData): string {
  const number = new Intl.NumberFormat('en', {
    notation: 'compact',
    maximumFractionDigits: 1,
  })
  const parts = [
    `${number.format(usage.input_tokens)} input`,
    `${number.format(usage.output_tokens)} output`,
  ]
  if (usage.estimated_cost_usd !== null) {
    parts.push(`~$${usage.estimated_cost_usd.toFixed(4)}`)
  }
  return parts.join(' · ')
}

function isTurnUsageData(value: unknown): value is TurnUsageData {
  if (typeof value !== 'object' || value === null) {
    return false
  }
  const record = value as Record<string, unknown>
  return (
    typeof record.input_tokens === 'number' &&
    record.input_tokens >= 0 &&
    typeof record.output_tokens === 'number' &&
    record.output_tokens >= 0 &&
    (record.estimated_cost_usd === null ||
      (typeof record.estimated_cost_usd === 'number' &&
        record.estimated_cost_usd >= 0))
  )
}
```

- [ ] **Step 5: Run the frontend tests**

Run:

```bash
cd frontend
npm test
```

Expected: PASS.

- [ ] **Step 6: Add the usage footer component**

Create `frontend/src/components/chat/UsageFooter.tsx`:

```tsx
import { formatUsageFooter, type TurnUsageData } from '@/lib/chat'

interface UsageFooterProps {
  usage: TurnUsageData
}

export function UsageFooter({ usage }: UsageFooterProps) {
  return (
    <p className="text-muted-foreground mt-3 border-t border-border/60 pt-2 text-xs tabular-nums">
      {formatUsageFooter(usage)}
    </p>
  )
}
```

- [ ] **Step 7: Render usage beneath citations**

In `frontend/src/components/chat/MessageList.tsx`:

```typescript
import { UsageFooter } from '@/components/chat/UsageFooter'
```

Import `messageUsage` from `@/lib/chat`. Inside the message map:

```typescript
const usage = messageUsage(message)
```

After `SourceCitations`, render:

```tsx
{!isUser &&
  !showProgress &&
  !showLiveActivity &&
  usage !== undefined && <UsageFooter usage={usage} />}
```

- [ ] **Step 8: Document the footer**

In `docs/guides/api-cost-optimization.md`, after the pricing configuration,
add:

```markdown
Completed assistant messages show a compact usage footer with input tokens,
output tokens, and estimated USD cost. Cost is omitted when any model used by
the turn lacks an exact entry in `CHAT_MODEL_PRICES`.
```

- [ ] **Step 9: Verify frontend tests, types, build, and lint**

Run:

```bash
cd frontend
npm test
npm run build
npm run lint
```

Expected: all commands exit 0.

- [ ] **Step 10: Verify the live UI**

With the backend and frontend dev servers running:

1. Send a chat question.
2. Confirm one footer appears below citations after the answer completes.
3. Confirm the footer matches the backend `chat.turn_complete` token totals and
   `estimated_cost_usd`.
4. Reload the page and confirm the same footer remains.
5. Load an older message without `data-usage` and confirm no empty footer
   appears.

- [ ] **Step 11: Run the full regression**

```bash
cd backend
uv run pytest -m "not integration" -q
uv run ruff check app tests

cd ../frontend
npm test
npm run build
npm run lint
```

Expected: all commands exit 0.

- [ ] **Step 12: Commit the frontend footer**

```bash
git add frontend/package.json frontend/src/lib/chat.ts \
  frontend/src/lib/chat.test.ts frontend/src/components/chat/UsageFooter.tsx \
  frontend/src/components/chat/MessageList.tsx \
  docs/guides/api-cost-optimization.md
git commit -m "feat: show chat usage below answers"
```


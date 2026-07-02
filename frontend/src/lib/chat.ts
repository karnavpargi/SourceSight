import type { UIMessage } from 'ai'

import type { MessageSummary } from '@/lib/api'

export interface CitationData {
  citation_index: number
  chunk_id: string
  excerpt: string
}

export interface SourcePassageData {
  chunk_id: string
  document_id: string
  chunk_index: number
  content: string
  section: string | null
  page: number | null
  ticker: string
  company_name: string | null
  form_type: string
  fiscal_year: number
  accession_number: string
  filing_date: string
  report_date: string | null
  source_url: string
  score: number
  is_neighbor: boolean
}

export interface ProgressData {
  label: string
  phase: 'running' | 'complete'
}

export interface TurnActivityData {
  step_id: string
  kind: string
  phase: 'start' | 'update' | 'end'
  label: string
  detail?: string | null
  order: number
}

export type ActivityStepStatus = 'running' | 'complete'

export interface ActivityTimelineStep {
  id: string
  kind: string
  label: string
  detail: string | null
  status: ActivityStepStatus
  order: number
}

export type SourceSightUIMessage = UIMessage<
  unknown,
  {
    citation: CitationData
    'source-passage': SourcePassageData
    progress: ProgressData
    activity: TurnActivityData
  }
>

export function toUIMessage(message: MessageSummary): SourceSightUIMessage {
  if (message.message_data) {
    return {
      ...(message.message_data as unknown as SourceSightUIMessage),
      // Persisted rows each have a unique DB id; client ids in message_data can repeat
      // when a stream request is retried after the user message was already saved.
      id: message.id,
    }
  }

  return {
    id: message.id,
    role: message.role === 'assistant' ? 'assistant' : 'user',
    parts: [{ type: 'text', text: message.content }],
  }
}

export function dedupeMessagesById<T extends { id: string }>(messages: T[]): T[] {
  const lastIndexById = new Map<string, number>()
  messages.forEach((message, index) => {
    lastIndexById.set(message.id, index)
  })

  return messages.filter((message, index) => lastIndexById.get(message.id) === index)
}

export function messageText(message: UIMessage): string {
  return message.parts
    .filter((part): part is Extract<typeof part, { type: 'text' }> => part.type === 'text')
    .map((part) => part.text)
    .join('')
}

export function messageProgress(message: UIMessage): ProgressData | null {
  let latest: ProgressData | null = null

  for (const part of message.parts) {
    if (part.type !== 'data-progress') {
      continue
    }
    if (!isProgressData(part.data)) {
      continue
    }
    latest = part.data
  }

  return latest
}

export function messageActivitySteps(message: UIMessage): ActivityTimelineStep[] {
  const steps = new Map<string, ActivityTimelineStep>()

  for (const part of message.parts) {
    if (part.type !== 'data-activity') {
      continue
    }
    if (!isTurnActivityData(part.data)) {
      continue
    }

    const data = part.data
    const existing = steps.get(data.step_id)
    const base: ActivityTimelineStep = existing ?? {
      id: data.step_id,
      kind: data.kind,
      label: data.label,
      detail: data.detail ?? null,
      status: 'running',
      order: data.order,
    }

    if (data.phase === 'start') {
      steps.set(data.step_id, {
        ...base,
        kind: data.kind,
        label: data.label,
        detail: data.detail ?? base.detail,
        status: 'running',
        order: data.order,
      })
      continue
    }

    if (data.phase === 'update') {
      steps.set(data.step_id, {
        ...base,
        kind: data.kind || base.kind,
        label: data.label || base.label,
        detail: data.detail ?? base.detail,
        status: 'running',
        order: Math.max(base.order, data.order),
      })
      continue
    }

    if (data.phase === 'end') {
      steps.set(data.step_id, {
        ...base,
        kind: data.kind || base.kind,
        label: data.label || base.label,
        detail: data.detail ?? base.detail,
        status: 'complete',
        order: Math.max(base.order, data.order),
      })
      continue
    }
  }

  return Array.from(steps.values())
    .filter((step) => step.label.trim().length > 0)
    .sort((left, right) => left.order - right.order)
}

export interface GroupedActivityStep {
  id: string
  kind: string
  label: string
  detail: string | null
  status: ActivityStepStatus
  order: number
  count: number
  members: ActivityTimelineStep[]
}

const KIND_LABELS: Record<string, string> = {
  thinking: 'Thinking',
  search_filings: 'Searching filings',
  read_chunk: 'Reading passage',
  read_surrounding_chunks: 'Reading surrounding context',
  validate: 'Validating sources',
  save: 'Saving answer',
}

function defaultKindLabel(kind: string): string {
  return KIND_LABELS[kind] ?? kind.replaceAll('_', ' ')
}

function parseGroupCount(label: string): number {
  const match = / ×(\d+)$/.exec(label)
  if (!match) {
    return 1
  }
  return Number.parseInt(match[1], 10)
}

export function groupActivitySteps(steps: ActivityTimelineStep[]): GroupedActivityStep[] {
  const groups: GroupedActivityStep[] = []

  for (const step of steps) {
    const previous = groups.at(-1)
    if (
      previous &&
      previous.kind === step.kind &&
      previous.status === 'complete' &&
      step.status === 'complete'
    ) {
      const count = previous.count + 1
      previous.count = count
      previous.members.push(step)
      previous.order = Math.max(previous.order, step.order)
      previous.label = `${defaultKindLabel(step.kind)} ×${count}`
      if (step.detail && !previous.detail) {
        previous.detail = step.detail
      }
      continue
    }

    const count = parseGroupCount(step.label)
    groups.push({
      id: `${step.kind}-${step.id}`,
      kind: step.kind,
      label: count > 1 ? step.label : step.label || defaultKindLabel(step.kind),
      detail: step.detail,
      status: step.status,
      order: step.order,
      count: Math.max(count, 1),
      members: [step],
    })
  }

  return groups
}

const VISIBLE_COMPLETED_GROUPS = 2

export function compactActivitySteps(steps: ActivityTimelineStep[]): {
  grouped: GroupedActivityStep[]
  hiddenCount: number
  running: GroupedActivityStep | null
} {
  const groups = groupActivitySteps(steps)
  if (groups.length === 0) {
    return { grouped: [], hiddenCount: 0, running: null }
  }

  const running = groups.find((group) => group.status === 'running') ?? null
  const completed = groups.filter((group) => group.status === 'complete')

  if (running === null) {
    const visible = completed.slice(-VISIBLE_COMPLETED_GROUPS)
    return {
      grouped: visible,
      hiddenCount: Math.max(0, completed.length - visible.length),
      running: null,
    }
  }

  const recentCompleted = completed.slice(-VISIBLE_COMPLETED_GROUPS)
  return {
    grouped: recentCompleted,
    hiddenCount: Math.max(0, completed.length - recentCompleted.length),
    running,
  }
}

export function hasActivityHistory(message: UIMessage): boolean {
  return message.role === 'assistant' && messageActivitySteps(message).length > 0
}

export function shouldShowActivityTimeline(
  message: UIMessage,
  {
    streaming,
    isActiveAssistantMessage,
  }: {
    streaming: boolean
    isActiveAssistantMessage: boolean
  },
): boolean {
  if (!hasActivityHistory(message)) {
    return false
  }

  if (isActiveAssistantMessage && streaming && messageText(message).length === 0) {
    return true
  }

  return false
}

export function shouldShowMessageProgress(
  message: UIMessage,
  {
    streaming,
    isActiveAssistantMessage,
  }: {
    streaming: boolean
    isActiveAssistantMessage: boolean
  },
): boolean {
  if (!isActiveAssistantMessage || !streaming || message.role !== 'assistant') {
    return false
  }

  const progress = messageProgress(message)
  if (progress === null || progress.phase !== 'running') {
    return false
  }

  return messageText(message).length === 0
}

export function messageCitations(message: UIMessage): CitationData[] {
  const citations: CitationData[] = []

  for (const part of message.parts) {
    if (part.type !== 'data-citation') {
      continue
    }
    if (!isCitationData(part.data)) {
      continue
    }
    citations.push(part.data)
  }

  return citations.sort((left, right) => left.citation_index - right.citation_index)
}

export function messageSourcePassages(message: UIMessage): SourcePassageData[] {
  const passages: SourcePassageData[] = []

  for (const part of message.parts) {
    if (part.type !== 'data-source-passage') {
      continue
    }
    if (!isSourcePassageData(part.data)) {
      continue
    }
    passages.push(part.data)
  }

  return passages
}

function isProgressData(value: unknown): value is ProgressData {
  if (typeof value !== 'object' || value === null) {
    return false
  }

  const record = value as Record<string, unknown>
  return (
    typeof record.label === 'string' &&
    (record.phase === 'running' || record.phase === 'complete')
  )
}

function isTurnActivityData(value: unknown): value is TurnActivityData {
  if (typeof value !== 'object' || value === null) {
    return false
  }

  const record = value as Record<string, unknown>
  return (
    typeof record.step_id === 'string' &&
    typeof record.kind === 'string' &&
    (record.phase === 'start' || record.phase === 'update' || record.phase === 'end') &&
    typeof record.label === 'string' &&
    typeof record.order === 'number'
  )
}

function isCitationData(value: unknown): value is CitationData {
  if (typeof value !== 'object' || value === null) {
    return false
  }

  const record = value as Record<string, unknown>
  return (
    typeof record.citation_index === 'number' &&
    typeof record.chunk_id === 'string' &&
    typeof record.excerpt === 'string'
  )
}

function isSourcePassageData(value: unknown): value is SourcePassageData {
  if (typeof value !== 'object' || value === null) {
    return false
  }

  const record = value as Record<string, unknown>
  return (
    typeof record.chunk_id === 'string' &&
    typeof record.ticker === 'string' &&
    typeof record.form_type === 'string' &&
    typeof record.fiscal_year === 'number' &&
    typeof record.content === 'string' &&
    typeof record.source_url === 'string'
  )
}

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

export type SourceSightUIMessage = UIMessage<
  unknown,
  {
    citation: CitationData
    'source-passage': SourcePassageData
  }
>

export function toUIMessage(message: MessageSummary): SourceSightUIMessage {
  if (message.message_data) {
    return message.message_data as unknown as SourceSightUIMessage
  }

  return {
    id: message.id,
    role: message.role === 'assistant' ? 'assistant' : 'user',
    parts: [{ type: 'text', text: message.content }],
  }
}

export function messageText(message: UIMessage): string {
  return message.parts
    .filter((part): part is Extract<typeof part, { type: 'text' }> => part.type === 'text')
    .map((part) => part.text)
    .join('')
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

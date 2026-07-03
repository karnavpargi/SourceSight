import type { CitationData, SourcePassageData } from '@/lib/chat'
import { normalizePassageMarkdown } from '@/lib/passage-text'

export interface SelectedCitation {
  citation: CitationData
  passage: SourcePassageData | null
}

export function resolveSelectedCitation(
  citations: CitationData[],
  passages: SourcePassageData[],
  citationIndex: number,
): SelectedCitation | null {
  const citation = citations.find((item) => item.citation_index === citationIndex)
  if (!citation) {
    return null
  }

  const passage =
    passages.find((item) => item.chunk_id === citation.chunk_id) ?? null

  return { citation, passage }
}

export function passagePreview(content: string, maxLength = 120): string {
  const normalized = normalizePassageMarkdown(content).replace(/\s+/g, ' ').trim()
  if (normalized.length <= maxLength) {
    return normalized
  }
  return `${normalized.slice(0, maxLength - 1)}…`
}

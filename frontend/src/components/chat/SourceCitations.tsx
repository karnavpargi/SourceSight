import { ExternalLink, FileText } from 'lucide-react'

import { MarkdownContent } from '@/components/chat/MarkdownContent'
import type { CitationData, SourcePassageData } from '@/lib/chat'
import { normalizePassageMarkdown } from '@/lib/passage-text'

interface SourceCitationsProps {
  citations: CitationData[]
  passages: SourcePassageData[]
  onCitationClick?: (citationIndex: number) => void
}

const filingDateFormatter = new Intl.DateTimeFormat(undefined, {
  year: 'numeric',
  month: 'short',
  day: 'numeric',
})

export function SourceCitations({
  citations,
  passages,
  onCitationClick,
}: SourceCitationsProps) {
  if (citations.length === 0) {
    return null
  }

  const passageByChunkId = new Map(passages.map((passage) => [passage.chunk_id, passage]))

  return (
    <div className="mt-4 space-y-3">
      <p className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
        Sources
      </p>
      {citations.map((citation) => {
        const passage = passageByChunkId.get(citation.chunk_id)

        return (
          <button
            key={`${citation.citation_index}-${citation.chunk_id}`}
            type="button"
            className="border-primary/20 bg-primary/5 hover:bg-primary/10 w-full cursor-pointer rounded-xl border p-3 text-left transition-colors duration-200"
            onClick={() => onCitationClick?.(citation.citation_index)}
          >
            <div className="flex items-start gap-3">
              <div className="bg-primary/10 text-primary flex size-8 shrink-0 items-center justify-center rounded-lg text-xs font-semibold">
                [{citation.citation_index}]
              </div>
              <div className="min-w-0 flex-1 space-y-2">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                  <p className="text-sm font-medium">
                    {passage?.ticker ?? 'Filing'}
                    {passage ? ` · ${passage.form_type} FY${passage.fiscal_year}` : ''}
                  </p>
                  {passage?.company_name && (
                    <p className="text-muted-foreground text-xs">{passage.company_name}</p>
                  )}
                </div>

                {passage?.section && (
                  <p className="text-muted-foreground text-xs">{passage.section}</p>
                )}

                <blockquote className="border-primary/30 text-foreground border-l-2 pl-3 text-xs leading-relaxed [&_table]:text-[11px]">
                  <MarkdownContent
                    content={normalizePassageMarkdown(citation.excerpt)}
                  />
                </blockquote>

                {passage && (
                  <div className="text-muted-foreground flex flex-wrap items-center gap-3 text-xs">
                    <span className="inline-flex items-center gap-1">
                      <FileText className="size-3.5" strokeWidth={2} />
                      Filed {formatFilingDate(passage.filing_date)}
                    </span>
                    <span className="text-primary inline-flex items-center gap-1 font-medium">
                      View source
                      <ExternalLink className="size-3.5" strokeWidth={2} />
                    </span>
                  </div>
                )}
              </div>
            </div>
          </button>
        )
      })}
    </div>
  )
}

function formatFilingDate(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return filingDateFormatter.format(date)
}

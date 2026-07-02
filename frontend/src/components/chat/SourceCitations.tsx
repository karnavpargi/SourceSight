import { ExternalLink, FileText } from 'lucide-react'

import type { CitationData, SourcePassageData } from '@/lib/chat'

interface SourceCitationsProps {
  citations: CitationData[]
  passages: SourcePassageData[]
}

const filingDateFormatter = new Intl.DateTimeFormat(undefined, {
  year: 'numeric',
  month: 'short',
  day: 'numeric',
})

export function SourceCitations({ citations, passages }: SourceCitationsProps) {
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
          <article
            key={`${citation.citation_index}-${citation.chunk_id}`}
            className="border-primary/20 bg-primary/5 rounded-xl border p-3"
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

                <blockquote className="border-primary/30 text-foreground border-l-2 pl-3 text-xs leading-relaxed">
                  {citation.excerpt}
                </blockquote>

                {passage && (
                  <details className="text-xs">
                    <summary className="text-primary cursor-pointer font-medium">
                      View full passage
                    </summary>
                    <p className="text-muted-foreground mt-2 whitespace-pre-wrap leading-relaxed">
                      {passage.content}
                    </p>
                    <div className="text-muted-foreground mt-2 flex flex-wrap items-center gap-3">
                      <span className="inline-flex items-center gap-1">
                        <FileText className="size-3.5" strokeWidth={2} />
                        Filed {formatFilingDate(passage.filing_date)}
                      </span>
                      <a
                        href={passage.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary inline-flex items-center gap-1 hover:underline"
                      >
                        Open filing
                        <ExternalLink className="size-3.5" strokeWidth={2} />
                      </a>
                    </div>
                  </details>
                )}
              </div>
            </div>
          </article>
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

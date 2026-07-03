import { ExternalLink, FileText } from 'lucide-react'

import { MarkdownContent } from '@/components/chat/MarkdownContent'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import type { SelectedCitation } from '@/lib/citations'
import { normalizePassageMarkdown } from '@/lib/passage-text'

interface SourcePassageDrawerProps {
  open: boolean
  selection: SelectedCitation | null
  onOpenChange: (open: boolean) => void
}

const filingDateFormatter = new Intl.DateTimeFormat(undefined, {
  year: 'numeric',
  month: 'short',
  day: 'numeric',
})

export function SourcePassageDrawer({
  open,
  selection,
  onOpenChange,
}: SourcePassageDrawerProps) {
  const passage = selection?.passage ?? null
  const citation = selection?.citation ?? null

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="border-glass-border bg-background/95 w-full gap-0 p-0 sm:max-w-lg"
      >
        <SheetHeader className="border-border/60 border-b px-6 py-5 text-left">
          <SheetTitle className="font-heading text-base">
            {passage
              ? `${passage.ticker} · ${passage.form_type} FY${passage.fiscal_year}`
              : 'Source passage'}
          </SheetTitle>
          <SheetDescription>
            {passage?.company_name ?? 'Retrieved filing excerpt'}
            {citation ? ` · Citation [${citation.citation_index}]` : ''}
          </SheetDescription>
        </SheetHeader>

        <div className="flex-1 space-y-5 overflow-y-auto px-6 py-5">
          {passage?.section && (
            <p className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
              {passage.section}
            </p>
          )}

          {citation && (
            <blockquote className="border-primary/30 bg-primary/5 rounded-xl border-l-4 p-4 text-sm leading-relaxed">
              <p className="text-muted-foreground mb-2 text-xs font-medium tracking-wide uppercase">
                Cited excerpt
              </p>
              <MarkdownContent
                content={normalizePassageMarkdown(citation.excerpt)}
              />
            </blockquote>
          )}

          {passage ? (
            <div className="space-y-3">
              <p className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
                Full passage
              </p>
              <div className="glass-panel rounded-2xl p-4 text-sm leading-relaxed [&_table]:text-xs">
                <MarkdownContent
                  content={normalizePassageMarkdown(passage.content)}
                />
              </div>

              <dl className="text-muted-foreground grid grid-cols-[auto_1fr] gap-x-3 gap-y-2 text-xs">
                <dt className="font-medium">Accession</dt>
                <dd className="font-mono">{passage.accession_number}</dd>
                <dt className="font-medium">Filed</dt>
                <dd>{formatFilingDate(passage.filing_date)}</dd>
                {passage.report_date && (
                  <>
                    <dt className="font-medium">Report date</dt>
                    <dd>{formatFilingDate(passage.report_date)}</dd>
                  </>
                )}
              </dl>

              <a
                href={passage.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="bg-primary text-primary-foreground hover:bg-primary/90 inline-flex cursor-pointer items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors duration-200"
              >
                <FileText className="size-4" strokeWidth={2} />
                Open original filing
                <ExternalLink className="size-3.5" strokeWidth={2} />
              </a>
            </div>
          ) : (
            <p className="text-muted-foreground text-sm">
              Source passage metadata is not available for this citation.
            </p>
          )}
        </div>
      </SheetContent>
    </Sheet>
  )
}

function formatFilingDate(value: string): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return filingDateFormatter.format(date)
}

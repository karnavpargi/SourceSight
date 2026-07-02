import { FileText } from 'lucide-react'

export function CitationPlaceholder() {
  return (
    <div className="border-primary/20 bg-primary/5 rounded-xl border border-dashed p-3">
      <div className="flex items-start gap-3">
        <div className="bg-primary/10 text-primary flex size-8 shrink-0 items-center justify-center rounded-lg">
          <FileText className="size-4" strokeWidth={2} />
        </div>
        <div>
          <p className="text-sm font-medium">Source citations</p>
          <p className="text-muted-foreground mt-1 text-xs leading-relaxed">
            Passages from indexed 10-K and 10-Q filings will appear here once
            retrieval is wired in.
          </p>
        </div>
      </div>
    </div>
  )
}

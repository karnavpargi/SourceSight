import { AlertTriangle } from 'lucide-react'

import type { CorpusStatus } from '@/lib/api'

interface CorpusStatusBannerProps {
  status: CorpusStatus | null
}

export function CorpusStatusBanner({ status }: CorpusStatusBannerProps) {
  if (!status || status.ready) {
    return null
  }

  return (
    <div
      className="border-amber-500/30 bg-amber-500/10 text-foreground flex items-start gap-3 border-b px-4 py-3 text-sm"
      role="status"
    >
      <AlertTriangle
        className="mt-0.5 size-4 shrink-0 text-amber-500"
        strokeWidth={2}
      />
      <div className="space-y-1">
        <p className="font-medium">No filings ingested yet</p>
        <p className="text-muted-foreground leading-relaxed">
          Run ingestion before asking research questions:{' '}
          <code className="bg-secondary/60 rounded px-1.5 py-0.5 font-mono text-xs">
            uv run python -m ingest.run
          </code>
        </p>
      </div>
    </div>
  )
}

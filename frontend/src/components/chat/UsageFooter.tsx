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

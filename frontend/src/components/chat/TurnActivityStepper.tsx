import { useState } from 'react'
import {
  Check,
  ChevronDown,
  ChevronUp,
  FileText,
  Loader2,
  Save,
  Search,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'

import type { ActivityTimelineStep, GroupedActivityStep } from '@/lib/chat'
import { groupActivitySteps } from '@/lib/chat'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface TurnActivityStepperProps {
  steps: ActivityTimelineStep[]
  grouped?: GroupedActivityStep[]
  hiddenCount?: number
  running?: GroupedActivityStep | null
  stickyActive?: boolean
  defaultExpanded?: boolean
  className?: string
}

function stepIcon(kind: string) {
  switch (kind) {
    case 'search_filings':
      return Search
    case 'read_chunk':
    case 'read_surrounding_chunks':
      return FileText
    case 'validate':
      return ShieldCheck
    case 'save':
      return Save
    case 'thinking':
      return Sparkles
    default:
      return Sparkles
  }
}

function GroupRow({
  group,
  expanded,
}: {
  group: GroupedActivityStep
  expanded: boolean
}) {
  const Icon = stepIcon(group.kind)
  const isRunning = group.status === 'running'

  return (
    <li
      className="flex items-start gap-2.5 text-sm"
      aria-current={isRunning ? 'step' : undefined}
    >
      <span
        className={cn(
          'mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full',
          isRunning
            ? 'bg-brand-purple/15 text-brand-purple'
            : 'bg-primary/10 text-primary',
        )}
      >
        {isRunning ? (
          <Loader2 className="size-3 animate-spin" strokeWidth={2.5} />
        ) : (
          <Check className="size-3" strokeWidth={2.5} />
        )}
      </span>

      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <Icon className="text-muted-foreground size-3.5 shrink-0" strokeWidth={2} />
          <span
            className={cn(
              'font-medium',
              isRunning ? 'text-foreground' : 'text-muted-foreground',
            )}
          >
            {group.label}
          </span>
        </div>
        {group.detail ? (
          <p className="text-muted-foreground mt-0.5 line-clamp-2 text-xs">{group.detail}</p>
        ) : null}
        {expanded && group.count > 1 ? (
          <ul className="text-muted-foreground mt-2 space-y-1 border-l pl-3 text-xs">
            {group.members.map((member) => (
              <li key={member.id} className="truncate">
                {member.detail ?? member.label}
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </li>
  )
}

export function TurnActivityStepper({
  steps,
  grouped,
  hiddenCount = 0,
  running = null,
  stickyActive = false,
  defaultExpanded = false,
  className,
}: TurnActivityStepperProps) {
  const [expanded, setExpanded] = useState(defaultExpanded)
  const allGroups = grouped ?? groupActivitySteps(steps)
  const completedGroups = allGroups.filter((group) => group.status === 'complete')
  const visibleGroups = expanded
    ? completedGroups
    : grouped ?? completedGroups.slice(-2)
  const resolvedHiddenCount = expanded
    ? 0
    : hiddenCount > 0
      ? hiddenCount
      : Math.max(0, completedGroups.length - visibleGroups.length)
  const canExpand = completedGroups.length > visibleGroups.length || resolvedHiddenCount > 0

  if (allGroups.length === 0 && running === null) {
    return null
  }

  return (
    <div className={cn('space-y-2', className)}>
      {canExpand ? (
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="text-muted-foreground hover:text-foreground h-7 cursor-pointer px-2 text-xs"
          onClick={() => {
            setExpanded((value) => !value)
          }}
        >
          {expanded ? (
            <>
              <ChevronUp className="size-3.5" strokeWidth={2} />
              Hide steps
            </>
          ) : (
            <>
              <ChevronDown className="size-3.5" strokeWidth={2} />
              Show all {completedGroups.length + (running ? 1 : 0)} steps
            </>
          )}
        </Button>
      ) : null}

      <div
        className={cn(
          stickyActive && running ? 'relative max-h-52 overflow-y-auto pr-1' : undefined,
        )}
      >
        <ol className="space-y-2" aria-label="Assistant activity">
          {visibleGroups.map((group) => (
            <GroupRow key={group.id} group={group} expanded={expanded} />
          ))}
        </ol>

        {stickyActive && running ? (
          <div className="border-border/60 bg-card/95 sticky bottom-0 mt-2 border-t pt-2 backdrop-blur-sm">
            <ol aria-label="Current assistant step">
              <GroupRow group={running} expanded={expanded} />
            </ol>
          </div>
        ) : running && !stickyActive ? (
          <ol className="mt-2 space-y-2" aria-label="Current assistant step">
            <GroupRow group={running} expanded={expanded} />
          </ol>
        ) : null}
      </div>

      {!expanded && resolvedHiddenCount > 0 ? (
        <p className="text-muted-foreground text-xs">
          {resolvedHiddenCount} earlier step{resolvedHiddenCount === 1 ? '' : 's'} completed
        </p>
      ) : null}
    </div>
  )
}

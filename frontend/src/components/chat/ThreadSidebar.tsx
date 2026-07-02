import { Loader2, LogOut, MessageSquarePlus, Sparkles } from 'lucide-react'

import type { ThreadSummary } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'

export interface ThreadSidebarContentProps {
  threads: ThreadSummary[]
  activeThreadId: string | null
  loading: boolean
  creating: boolean
  error: string | null
  onSelectThread: (threadId: string) => void
  onCreateThread: () => void
  onSignOut: () => void
}

function formatThreadDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(value))
}

export function ThreadSidebarContent({
  threads,
  activeThreadId,
  loading,
  creating,
  error,
  onSelectThread,
  onCreateThread,
  onSignOut,
}: ThreadSidebarContentProps) {
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-sidebar-border flex items-center justify-between border-b p-4">
        <div className="flex items-center gap-3">
          <div className="bg-primary/15 text-primary flex size-9 items-center justify-center rounded-lg">
            <Sparkles className="size-4" strokeWidth={2} />
          </div>
          <div>
            <p className="font-heading text-sm font-semibold tracking-tight">
              SourceSight
            </p>
            <p className="text-muted-foreground text-xs">Research threads</p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="icon-sm"
          className="cursor-pointer transition-colors duration-200"
          aria-label="Sign out"
          onClick={() => void onSignOut()}
        >
          <LogOut className="size-4" strokeWidth={2} />
        </Button>
      </div>

      <div className="border-sidebar-border border-b p-4">
        <Button
          className="bg-cta text-cta-foreground hover:bg-cta/90 w-full cursor-pointer transition-colors duration-200"
          disabled={creating}
          onClick={() => onCreateThread()}
        >
          {creating ? (
            <>
              <Loader2 className="size-4 animate-spin" strokeWidth={2} />
              Creating…
            </>
          ) : (
            <>
              <MessageSquarePlus className="size-4" strokeWidth={2} />
              New analysis
            </>
          )}
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {loading ? (
          <div className="space-y-2">
            <Skeleton className="h-16 w-full rounded-xl" />
            <Skeleton className="h-16 w-full rounded-xl" />
            <Skeleton className="h-16 w-full rounded-xl" />
          </div>
        ) : threads.length === 0 ? (
          <p className="text-muted-foreground p-3 text-sm leading-relaxed">
            No threads yet. Start a new analysis to query indexed filings.
          </p>
        ) : (
          <ul className="space-y-1">
            {threads.map((thread) => {
              const active = thread.id === activeThreadId
              return (
                <li key={thread.id}>
                  <button
                    type="button"
                    className={`w-full cursor-pointer rounded-xl border px-3 py-3 text-left transition-colors duration-200 ${
                      active
                        ? 'border-primary/40 bg-primary/10'
                        : 'hover:bg-sidebar-accent border-transparent'
                    }`}
                    onClick={() => onSelectThread(thread.id)}
                  >
                    <p className="truncate text-sm font-medium">{thread.title}</p>
                    <p className="text-muted-foreground mt-1 text-xs">
                      {formatThreadDate(thread.updated_at)}
                    </p>
                  </button>
                </li>
              )
            })}
          </ul>
        )}
      </div>

      {error && (
        <p
          className="text-destructive border-sidebar-border border-t p-4 text-sm"
          role="alert"
        >
          {error}
        </p>
      )}
    </div>
  )
}

interface ThreadSidebarProps extends ThreadSidebarContentProps {
  className?: string
}

export function ThreadSidebar({ className, ...props }: ThreadSidebarProps) {
  return (
    <aside
      className={cn(
        'border-sidebar-border bg-sidebar text-sidebar-foreground flex w-72 shrink-0 flex-col border-r backdrop-blur-xl',
        className,
      )}
    >
      <ThreadSidebarContent {...props} />
    </aside>
  )
}

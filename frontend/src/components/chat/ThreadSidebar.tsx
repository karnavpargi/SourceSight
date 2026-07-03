import { useState } from 'react'
import {
  Loader2,
  LogOut,
  MessageSquarePlus,
  MoreHorizontal,
  Pencil,
  Sparkles,
  Trash2,
} from 'lucide-react'

import type { ThreadSummary } from '@/lib/api'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
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
  onRenameThread: (threadId: string, title: string) => void
  onDeleteThread: (threadId: string) => void
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
  onRenameThread,
  onDeleteThread,
  onSignOut,
}: ThreadSidebarContentProps) {
  const [renameThreadId, setRenameThreadId] = useState<string | null>(null)
  const [renameTitle, setRenameTitle] = useState('')
  const [deleteThreadId, setDeleteThreadId] = useState<string | null>(null)

  const renameTarget = threads.find((thread) => thread.id === renameThreadId) ?? null
  const deleteTarget = threads.find((thread) => thread.id === deleteThreadId) ?? null

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
                  <div
                    className={`group flex items-stretch gap-1 rounded-xl border transition-colors duration-200 ${
                      active
                        ? 'border-primary/40 bg-primary/10'
                        : 'hover:bg-sidebar-accent border-transparent'
                    }`}
                  >
                    <button
                      type="button"
                      className="min-w-0 flex-1 cursor-pointer px-3 py-3 text-left"
                      onClick={() => onSelectThread(thread.id)}
                    >
                      <p className="truncate text-sm font-medium">{thread.title}</p>
                      <p className="text-muted-foreground mt-1 text-xs">
                        {formatThreadDate(thread.updated_at)}
                      </p>
                    </button>

                    <Popover>
                      <PopoverTrigger asChild>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon-sm"
                          className="my-2 mr-1 shrink-0 cursor-pointer opacity-100 transition-opacity duration-200 md:opacity-0 md:group-hover:opacity-100 md:group-focus-within:opacity-100"
                          aria-label={`Thread actions for ${thread.title}`}
                        >
                          <MoreHorizontal className="size-4" strokeWidth={2} />
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent align="end" className="w-44 p-1">
                        <button
                          type="button"
                          className="hover:bg-accent flex w-full cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors duration-200"
                          onClick={() => {
                            setRenameThreadId(thread.id)
                            setRenameTitle(thread.title)
                          }}
                        >
                          <Pencil className="size-4" strokeWidth={2} />
                          Rename
                        </button>
                        <button
                          type="button"
                          className="text-destructive hover:bg-destructive/10 flex w-full cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors duration-200"
                          onClick={() => setDeleteThreadId(thread.id)}
                        >
                          <Trash2 className="size-4" strokeWidth={2} />
                          Delete
                        </button>
                      </PopoverContent>
                    </Popover>
                  </div>
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

      <Dialog
        open={renameThreadId !== null}
        onOpenChange={(open) => {
          if (!open) {
            setRenameThreadId(null)
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename thread</DialogTitle>
            <DialogDescription>
              Give this analysis a title your team will recognize later.
            </DialogDescription>
          </DialogHeader>
          <Input
            value={renameTitle}
            maxLength={255}
            onChange={(event) => setRenameTitle(event.target.value)}
            aria-label="Thread title"
          />
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              className="cursor-pointer"
              onClick={() => setRenameThreadId(null)}
            >
              Cancel
            </Button>
            <Button
              type="button"
              className="cursor-pointer"
              disabled={renameTitle.trim().length === 0}
              onClick={() => {
                if (!renameThreadId) {
                  return
                }
                onRenameThread(renameThreadId, renameTitle.trim())
                setRenameThreadId(null)
              }}
            >
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={deleteThreadId !== null}
        onOpenChange={(open) => {
          if (!open) {
            setDeleteThreadId(null)
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete thread?</DialogTitle>
            <DialogDescription>
              {deleteTarget
                ? `“${deleteTarget.title}” and its messages will be removed permanently.`
                : 'This thread and its messages will be removed permanently.'}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              className="cursor-pointer"
              onClick={() => setDeleteThreadId(null)}
            >
              Cancel
            </Button>
            <Button
              type="button"
              variant="destructive"
              className="cursor-pointer"
              onClick={() => {
                if (!deleteThreadId) {
                  return
                }
                onDeleteThread(deleteThreadId)
                setDeleteThreadId(null)
              }}
            >
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
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
        'border-sidebar-border bg-sidebar text-sidebar-foreground flex w-72 shrink-0 flex-col border-r backdrop-blur-xl lg:w-80',
        className,
      )}
    >
      <ThreadSidebarContent {...props} />
    </aside>
  )
}

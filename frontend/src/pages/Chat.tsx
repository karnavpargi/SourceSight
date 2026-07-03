import type { UIMessage } from 'ai'
import { Menu, MessageSquarePlus, Sparkles } from 'lucide-react'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { toast } from 'sonner'

import { ChatPanel } from '@/components/chat/ChatPanel'
import { StarterPrompts } from '@/components/chat/StarterPrompts'
import {
  ThreadSidebar,
  ThreadSidebarContent,
} from '@/components/chat/ThreadSidebar'
import { CorpusStatusBanner } from '@/components/CorpusStatusBanner'
import { Button } from '@/components/ui/button'
import { Sheet, SheetContent } from '@/components/ui/sheet'
import { Skeleton } from '@/components/ui/skeleton'
import { api, type CorpusStatus, type ThreadSummary } from '@/lib/api'
import { useAuth } from '@/lib/auth'
import { toUIMessage } from '@/lib/chat'
import { stashPendingPrompt } from '@/lib/chat-prompts'
import { DEFAULT_THREAD_TITLE } from '@/lib/chat-threads'
import { ApiError, isNetworkError } from '@/lib/http'
import { registerSessionExpiredHandler } from '@/lib/session-expired'

function formatError(error: unknown): string {
  if (error instanceof ApiError) {
    if (isNetworkError(error)) {
      return 'Could not reach the API. Check that the backend is running and CORS is configured.'
    }
    return error.detail ?? error.message
  }

  if (error instanceof Error) {
    return error.message
  }

  return 'Something went wrong.'
}

export function Chat() {
  const navigate = useNavigate()
  const { threadId } = useParams()
  const { signOut } = useAuth()

  const [threads, setThreads] = useState<ThreadSummary[]>([])
  const [threadsLoading, setThreadsLoading] = useState(true)
  const [threadsError, setThreadsError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const [initialMessages, setInitialMessages] = useState<UIMessage[]>([])
  const [messagesLoading, setMessagesLoading] = useState(Boolean(threadId))
  const [messagesError, setMessagesError] = useState<string | null>(null)
  const [corpusStatus, setCorpusStatus] = useState<CorpusStatus | null>(null)

  const activeThread = useMemo(
    () => threads.find((thread) => thread.id === threadId) ?? null,
    [threadId, threads],
  )

  const loadThreads = useCallback(async () => {
    setThreadsLoading(true)
    setThreadsError(null)

    try {
      const nextThreads = await api.listThreads()
      setThreads(nextThreads)
    } catch (error) {
      setThreadsError(formatError(error))
    } finally {
      setThreadsLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true

    void (async () => {
      try {
        const nextThreads = await api.listThreads()
        if (active) {
          setThreads(nextThreads)
        }
      } catch (error) {
        if (active) {
          setThreadsError(formatError(error))
        }
      } finally {
        if (active) {
          setThreadsLoading(false)
        }
      }
    })()

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    return registerSessionExpiredHandler(() => {
      void (async () => {
        await signOut()
        toast.error('Your session expired. Please sign in again.')
        navigate('/sign-in', { replace: true })
      })()
    })
  }, [navigate, signOut])

  useEffect(() => {
    let active = true

    void (async () => {
      try {
        const status = await api.getCorpusStatus()
        if (active) {
          setCorpusStatus(status)
        }
      } catch {
        if (active) {
          setCorpusStatus(null)
        }
      }
    })()

    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (!threadId) {
      return
    }

    let active = true

    void (async () => {
      setMessagesLoading(true)
      setInitialMessages([])
      setMessagesError(null)

      try {
        const messages = await api.listThreadMessages(threadId)
        if (active) {
          setInitialMessages(messages.map(toUIMessage))
        }
      } catch (error) {
        if (active) {
          setMessagesError(formatError(error))
        }
      } finally {
        if (active) {
          setMessagesLoading(false)
        }
      }
    })()

    return () => {
      active = false
    }
  }, [threadId])

  async function handleCreateThread(options?: { initialPrompt?: string }) {
    setCreating(true)

    try {
      const thread = await api.createThread({ title: DEFAULT_THREAD_TITLE })
      if (options?.initialPrompt) {
        stashPendingPrompt(thread.id, options.initialPrompt)
      }
      setMessagesLoading(true)
      await loadThreads()
      navigate(`/chat/${thread.id}`)
      setSidebarOpen(false)
      toast.success('Thread created')
    } catch (error) {
      toast.error(formatError(error))
    } finally {
      setCreating(false)
    }
  }

  function handleSelectThread(id: string) {
    setMessagesLoading(true)
    setSidebarOpen(false)
    navigate(`/chat/${id}`)
  }

  async function handleRenameThread(threadId: string, title: string) {
    try {
      const updated = await api.updateThread(threadId, { title })
      setThreads((current) =>
        current.map((thread) => (thread.id === updated.id ? updated : thread)),
      )
      toast.success('Thread renamed')
    } catch (error) {
      toast.error(formatError(error))
    }
  }

  async function handleDeleteThread(deletedThreadId: string) {
    try {
      await api.deleteThread(deletedThreadId)
      setThreads((current) =>
        current.filter((thread) => thread.id !== deletedThreadId),
      )
      if (deletedThreadId === threadId) {
        navigate('/chat')
      }
      toast.success('Thread deleted')
    } catch (error) {
      toast.error(formatError(error))
    }
  }

  const sidebarProps = {
    threads,
    activeThreadId: threadId ?? null,
    loading: threadsLoading,
    creating,
    error: threadsError,
    onSelectThread: handleSelectThread,
    onCreateThread: () => void handleCreateThread(),
    onRenameThread: (id: string, title: string) => void handleRenameThread(id, title),
    onDeleteThread: (id: string) => void handleDeleteThread(id),
    onSignOut: async () => {
      await signOut()
      navigate('/sign-in', { replace: true })
    },
  }

  return (
    <div className="flex h-svh overflow-hidden">
      <ThreadSidebar {...sidebarProps} className="hidden md:flex" />

      <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
        <SheetContent
          side="left"
          className="border-sidebar-border bg-sidebar text-sidebar-foreground w-72 max-w-[85vw] gap-0 p-0 sm:max-w-sm"
        >
          <ThreadSidebarContent {...sidebarProps} />
        </SheetContent>
      </Sheet>

      <main className="flex min-w-0 flex-1 flex-col">
        <CorpusStatusBanner status={corpusStatus} />
        {!threadId ? (
          <>
            <div className="border-border/60 glass-panel flex items-center gap-3 border-b px-4 py-3 md:hidden">
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                className="shrink-0 cursor-pointer"
                aria-label="Open threads"
                onClick={() => setSidebarOpen(true)}
              >
                <Menu className="size-5" strokeWidth={2} />
              </Button>
              <p className="font-heading truncate text-sm font-semibold tracking-tight">
                SourceSight
              </p>
            </div>

            <div className="flex flex-1 flex-col items-center justify-center gap-6 p-6">
              <div className="glass-panel max-w-lg rounded-3xl p-8 text-center">
                <div className="bg-primary/15 text-primary mx-auto mb-4 flex size-12 items-center justify-center rounded-2xl">
                  <Sparkles className="size-6" strokeWidth={1.75} />
                </div>
                <h1 className="font-heading text-2xl font-semibold tracking-tight">
                  Select or create an analysis
                </h1>
                <p className="text-muted-foreground mt-2 text-sm leading-relaxed">
                  Your filing conversations live in the sidebar. Start a new thread
                  to query indexed 10-K and 10-Q documents with cited answers.
                </p>
                <Button
                  className="bg-cta text-cta-foreground hover:bg-cta/90 mt-6 cursor-pointer transition-colors duration-200"
                  onClick={() => void handleCreateThread()}
                  disabled={creating}
                >
                  <MessageSquarePlus className="size-4" strokeWidth={2} />
                  New analysis
                </Button>
              </div>

              <StarterPrompts
                className="max-w-lg"
                disabled={creating}
                onSelect={(prompt) => void handleCreateThread({ initialPrompt: prompt })}
              />
            </div>
          </>
        ) : messagesLoading ? (
          <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4 p-6">
            <Skeleton className="h-24 w-2/3 rounded-2xl" />
            <Skeleton className="h-32 w-3/4 self-end rounded-2xl" />
          </div>
        ) : messagesError ? (
          <div
            className="text-destructive flex flex-1 items-center justify-center p-6 text-sm"
            role="alert"
          >
            {messagesError}
          </div>
        ) : (
          <ChatPanel
            key={threadId}
            threadId={threadId}
            threadTitle={activeThread?.title ?? 'Analysis'}
            initialMessages={initialMessages}
            onOpenSidebar={() => setSidebarOpen(true)}
            onThreadsChange={loadThreads}
          />
        )}
      </main>
    </div>
  )
}

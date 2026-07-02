import type { UIMessage } from 'ai'
import { MessageSquarePlus, Sparkles } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { toast } from 'sonner'

import { ChatPanel } from '@/components/chat/ChatPanel'
import { ThreadSidebar } from '@/components/chat/ThreadSidebar'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { api, type ThreadSummary } from '@/lib/api'
import { useAuth } from '@/lib/auth'
import { toUIMessage } from '@/lib/chat'
import { ApiError, isNetworkError } from '@/lib/http'

const STARTER_PROMPTS = [
  'What was AWS operating income last quarter?',
  'Summarize Apple risk factors from the latest 10-K',
  'Compare NVDA and AMD revenue growth trends',
] as const

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

  const [initialMessages, setInitialMessages] = useState<UIMessage[]>([])
  const [messagesLoading, setMessagesLoading] = useState(Boolean(threadId))
  const [messagesError, setMessagesError] = useState<string | null>(null)

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
    if (!threadId) {
      return
    }

    let active = true

    void (async () => {
      try {
        const messages = await api.listThreadMessages(threadId)
        if (active) {
          setInitialMessages(messages.map(toUIMessage))
          setMessagesError(null)
        }
      } catch (error) {
        if (active) {
          setMessagesError(formatError(error))
          setInitialMessages([])
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

  async function handleCreateThread() {
    setCreating(true)

    try {
      const thread = await api.createThread({ title: 'New analysis' })
      setMessagesLoading(true)
      await loadThreads()
      navigate(`/chat/${thread.id}`)
      toast.success('Thread created')
    } catch (error) {
      toast.error(formatError(error))
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="flex h-svh overflow-hidden">
      <ThreadSidebar
        threads={threads}
        activeThreadId={threadId ?? null}
        loading={threadsLoading}
        creating={creating}
        error={threadsError}
        onSelectThread={(id) => {
          setMessagesLoading(true)
          navigate(`/chat/${id}`)
        }}
        onCreateThread={() => void handleCreateThread()}
        onSignOut={async () => {
          await signOut()
          navigate('/sign-in', { replace: true })
        }}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        {!threadId ? (
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

            <div className="max-w-lg space-y-3">
              <p className="text-muted-foreground text-center text-xs font-medium tracking-wide uppercase">
                Example prompts
              </p>
              <ul className="space-y-2">
                {STARTER_PROMPTS.map((prompt) => (
                  <li
                    key={prompt}
                    className="glass-panel text-muted-foreground cursor-default rounded-xl px-4 py-3 text-sm"
                  >
                    {prompt}
                  </li>
                ))}
              </ul>
            </div>
          </div>
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
            initialMessages={initialMessages}
          />
        )}
      </main>
    </div>
  )
}

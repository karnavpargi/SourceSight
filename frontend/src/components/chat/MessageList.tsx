import { useState } from 'react'
import type { UIMessage } from 'ai'
import { Bot, Loader2, UserRound } from 'lucide-react'

import { MarkdownContent } from '@/components/chat/MarkdownContent'
import { SourceCitations } from '@/components/chat/SourceCitations'
import { SourcePassageDrawer } from '@/components/chat/SourcePassageDrawer'
import { StarterPrompts } from '@/components/chat/StarterPrompts'
import { TurnActivityStepper } from '@/components/chat/TurnActivityStepper'
import { Skeleton } from '@/components/ui/skeleton'
import { useStickToBottom } from '@/hooks/useStickToBottom'
import {
  compactActivitySteps,
  dedupeMessagesById,
  hasActivityHistory,
  messageActivitySteps,
  messageCitations,
  messageProgress,
  messageSourcePassages,
  messageText,
  shouldShowActivityTimeline,
  shouldShowMessageProgress,
} from '@/lib/chat'
import { resolveSelectedCitation, type SelectedCitation } from '@/lib/citations'

interface MessageListProps {
  messages: UIMessage[]
  loading: boolean
  streaming: boolean
  onSelectPrompt?: (prompt: string) => void
  promptsDisabled?: boolean
}

export function MessageList({
  messages,
  loading,
  streaming,
  onSelectPrompt,
  promptsDisabled = false,
}: MessageListProps) {
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [selection, setSelection] = useState<SelectedCitation | null>(null)

  const visibleMessages = dedupeMessagesById(messages)
  const lastMessage = visibleMessages[visibleMessages.length - 1]
  const lastMessageText = lastMessage ? messageText(lastMessage) : ''
  const lastActivitySteps = lastMessage ? messageActivitySteps(lastMessage) : []
  const lastRunningStep = lastActivitySteps.find((step) => step.status === 'running')
  const scrollKey = [
    visibleMessages.length,
    lastMessageText.length,
    lastActivitySteps.length,
    lastRunningStep?.label ?? '',
    lastRunningStep?.detail ?? '',
    streaming,
  ].join(':')

  const { containerRef, handleScroll } = useStickToBottom(scrollKey)

  function openCitation(
    citations: ReturnType<typeof messageCitations>,
    passages: ReturnType<typeof messageSourcePassages>,
    citationIndex: number,
  ) {
    const nextSelection = resolveSelectedCitation(citations, passages, citationIndex)
    if (!nextSelection) {
      return
    }
    setSelection(nextSelection)
    setDrawerOpen(true)
  }

  if (loading) {
    return (
      <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4 p-4 sm:p-6">
        <Skeleton className="h-24 w-2/3 rounded-2xl" />
        <Skeleton className="h-32 w-3/4 self-end rounded-2xl" />
        <Skeleton className="h-24 w-2/3 rounded-2xl" />
      </div>
    )
  }

  if (messages.length === 0) {
    return (
      <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center gap-6 p-4 text-center sm:p-6">
        <div className="space-y-4">
          <div className="glass-panel rounded-2xl p-4">
            <Bot className="text-primary mx-auto size-8" strokeWidth={1.75} />
          </div>
          <div className="space-y-2">
            <p className="font-heading text-xl font-semibold tracking-tight">
              Start your analysis
            </p>
            <p className="text-muted-foreground max-w-md text-sm leading-relaxed">
              Ask about revenue, risk factors, or segment performance. Answers
              stream here with source citations.
            </p>
          </div>
        </div>

        {onSelectPrompt && (
          <StarterPrompts
            className="w-full max-w-md"
            disabled={promptsDisabled}
            onSelect={onSelectPrompt}
          />
        )}
      </div>
    )
  }

  const showStreamingPlaceholder =
    streaming && (!lastMessage || lastMessage.role === 'user')

  return (
    <>
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="mx-auto flex min-h-0 w-full max-w-3xl flex-1 flex-col gap-5 overflow-y-auto p-4 sm:p-6"
      >
        {visibleMessages.map((message) => {
          const isUser = message.role === 'user'
          const text = messageText(message)
          const progress = messageProgress(message)
          const activitySteps = messageActivitySteps(message)
          const activityView = compactActivitySteps(activitySteps)
          const citations = messageCitations(message)
          const passages = messageSourcePassages(message)
          const Icon = isUser ? UserRound : Bot
          const isActiveAssistantMessage =
            !isUser && streaming && message.id === lastMessage?.id
          const showLiveActivity = shouldShowActivityTimeline(message, {
            streaming,
            isActiveAssistantMessage,
          })
          const showProgress = shouldShowMessageProgress(message, {
            streaming,
            isActiveAssistantMessage,
          })
          const showPersistedActivity =
            !isUser &&
            !showLiveActivity &&
            hasActivityHistory(message) &&
            text.length > 0

          return (
            <div
              key={message.id}
              className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}
            >
              <div
                className={`flex size-8 shrink-0 items-center justify-center rounded-lg ${
                  isUser
                    ? 'bg-primary/15 text-primary'
                    : 'bg-brand-purple/15 text-brand-purple'
                }`}
              >
                <Icon className="size-4" strokeWidth={2} />
              </div>

              <div
                className={`max-w-[min(85%,42rem)] rounded-2xl border px-4 py-3 text-sm leading-relaxed ${
                  isUser
                    ? 'border-primary/20 bg-primary/10 text-foreground'
                    : 'glass-panel text-foreground'
                }`}
              >
                <p className="text-muted-foreground mb-2 text-xs font-medium tracking-wide uppercase">
                  {isUser ? 'You' : 'SourceSight'}
                </p>
                {showLiveActivity ? (
                  <TurnActivityStepper
                    steps={activitySteps}
                    grouped={activityView.grouped}
                    hiddenCount={activityView.hiddenCount}
                    running={activityView.running}
                    stickyActive
                  />
                ) : showProgress ? (
                  <div className="text-muted-foreground flex items-center gap-2">
                    <Loader2 className="size-4 shrink-0 animate-spin" strokeWidth={2} />
                    <span>{progress?.label}</span>
                  </div>
                ) : text.length > 0 ? (
                  isUser ? (
                    <p className="whitespace-pre-wrap">{text}</p>
                  ) : (
                    <MarkdownContent
                      content={text}
                      onCitationClick={(citationIndex) =>
                        openCitation(citations, passages, citationIndex)
                      }
                    />
                  )
                ) : isActiveAssistantMessage ? (
                  <div className="text-muted-foreground flex items-center gap-2">
                    <Loader2 className="size-4 shrink-0 animate-spin" strokeWidth={2} />
                    <span>Preparing answer…</span>
                  </div>
                ) : null}
                {showPersistedActivity ? (
                  <details className="border-border/60 mt-3 border-t pt-3">
                    <summary className="text-muted-foreground cursor-pointer text-xs font-medium tracking-wide uppercase">
                      Research steps
                    </summary>
                    <TurnActivityStepper
                      steps={activitySteps}
                      className="mt-3"
                    />
                  </details>
                ) : null}
                {!isUser && !showProgress && !showLiveActivity && (
                  <SourceCitations
                    citations={citations}
                    passages={passages}
                    onCitationClick={(citationIndex) =>
                      openCitation(citations, passages, citationIndex)
                    }
                  />
                )}
              </div>
            </div>
          )
        })}

        {showStreamingPlaceholder && (
          <div className="flex gap-3">
            <div className="bg-brand-purple/15 text-brand-purple flex size-8 shrink-0 items-center justify-center rounded-lg">
              <Bot className="size-4" strokeWidth={2} />
            </div>
            <div className="glass-panel rounded-2xl px-4 py-3">
              <div className="text-muted-foreground flex items-center gap-2">
                <Loader2 className="size-4 shrink-0 animate-spin" strokeWidth={2} />
                <span>Connecting...</span>
              </div>
            </div>
          </div>
        )}
      </div>

      <SourcePassageDrawer
        open={drawerOpen}
        selection={selection}
        onOpenChange={setDrawerOpen}
      />
    </>
  )
}

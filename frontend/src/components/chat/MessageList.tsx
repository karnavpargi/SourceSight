import type { UIMessage } from 'ai'
import { Bot, Loader2, UserRound } from 'lucide-react'

import { MarkdownContent } from '@/components/chat/MarkdownContent'
import { SourceCitations } from '@/components/chat/SourceCitations'
import { StarterPrompts } from '@/components/chat/StarterPrompts'
import { Skeleton } from '@/components/ui/skeleton'
import { useStickToBottom } from '@/hooks/useStickToBottom'
import {
  dedupeMessagesById,
  messageCitations,
  messageProgress,
  messageSourcePassages,
  messageText,
  shouldShowMessageProgress,
} from '@/lib/chat'

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
  const visibleMessages = dedupeMessagesById(messages)
  const lastMessage = visibleMessages[visibleMessages.length - 1]
  const lastMessageText = lastMessage ? messageText(lastMessage) : ''
  const scrollKey = `${visibleMessages.length}:${lastMessageText.length}:${streaming}`

  const { containerRef, handleScroll } = useStickToBottom(scrollKey)

  if (loading) {
    return (
      <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-4 p-6">
        <Skeleton className="h-24 w-2/3 rounded-2xl" />
        <Skeleton className="h-32 w-3/4 self-end rounded-2xl" />
        <Skeleton className="h-24 w-2/3 rounded-2xl" />
      </div>
    )
  }

  if (messages.length === 0) {
    return (
      <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center gap-6 p-6 text-center">
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
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-5 overflow-y-auto p-6"
    >
      {visibleMessages.map((message) => {
        const isUser = message.role === 'user'
        const text = messageText(message)
        const progress = messageProgress(message)
        const citations = messageCitations(message)
        const passages = messageSourcePassages(message)
        const Icon = isUser ? UserRound : Bot
        const isActiveAssistantMessage =
          !isUser && streaming && message.id === lastMessage?.id
        const showProgress = shouldShowMessageProgress(message, {
          streaming,
          isActiveAssistantMessage,
        })

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
              className={`max-w-[85%] rounded-2xl border px-4 py-3 text-sm leading-relaxed ${
                isUser
                  ? 'border-primary/20 bg-primary/10 text-foreground'
                  : 'glass-panel text-foreground'
              }`}
            >
              <p className="text-muted-foreground mb-2 text-xs font-medium tracking-wide uppercase">
                {isUser ? 'You' : 'SourceSight'}
              </p>
              {showProgress ? (
                <div className="text-muted-foreground flex items-center gap-2">
                  <Loader2 className="size-4 shrink-0 animate-spin" strokeWidth={2} />
                  <span>{progress?.label}</span>
                </div>
              ) : text.length > 0 ? (
                isUser ? (
                  <p className="whitespace-pre-wrap">{text}</p>
                ) : (
                  <MarkdownContent content={text} />
                )
              ) : isActiveAssistantMessage ? (
                <div className="text-muted-foreground flex items-center gap-2">
                  <Loader2 className="size-4 shrink-0 animate-spin" strokeWidth={2} />
                  <span>Preparing answer…</span>
                </div>
              ) : null}
              {!isUser && !showProgress && (
                <SourceCitations citations={citations} passages={passages} />
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
  )
}

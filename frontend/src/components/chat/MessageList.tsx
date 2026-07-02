import type { UIMessage } from 'ai'
import { Bot, UserRound } from 'lucide-react'

import { SourceCitations } from '@/components/chat/SourceCitations'
import { Skeleton } from '@/components/ui/skeleton'
import { messageCitations, messageSourcePassages, messageText } from '@/lib/chat'

interface MessageListProps {
  messages: UIMessage[]
  loading: boolean
  streaming: boolean
}

export function MessageList({
  messages,
  loading,
  streaming,
}: MessageListProps) {
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
      <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center gap-4 p-6 text-center">
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
    )
  }

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col gap-5 overflow-y-auto p-6">
      {messages.map((message) => {
        const isUser = message.role === 'user'
        const text = messageText(message)
        const citations = messageCitations(message)
        const passages = messageSourcePassages(message)
        const Icon = isUser ? UserRound : Bot

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
              <p className="whitespace-pre-wrap">{text}</p>
              {!isUser && (
                <SourceCitations citations={citations} passages={passages} />
              )}
            </div>
          </div>
        )
      })}

      {streaming && (
        <div className="flex gap-3">
          <div className="bg-brand-purple/15 text-brand-purple flex size-8 shrink-0 items-center justify-center rounded-lg">
            <Bot className="size-4" strokeWidth={2} />
          </div>
          <div className="glass-panel rounded-2xl px-4 py-3">
            <Skeleton className="h-4 w-48" />
          </div>
        </div>
      )}
    </div>
  )
}

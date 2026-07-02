import { useMemo, useState } from 'react'
import { useChat } from '@ai-sdk/react'
import { DefaultChatTransport, type UIMessage } from 'ai'
import { AlertCircle } from 'lucide-react'

import { ChatInput } from '@/components/chat/ChatInput'
import { MessageList } from '@/components/chat/MessageList'
import { env } from '@/lib/env'
import { api } from '@/lib/api'

interface ChatPanelProps {
  threadId: string
  initialMessages: UIMessage[]
}

export function ChatPanel({ threadId, initialMessages }: ChatPanelProps) {
  const [input, setInput] = useState('')

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: `${env.apiBaseUrl}/chat/stream`,
        headers: async () => {
          const token = await api.getAccessToken()
          const headers: Record<string, string> = {}
          if (token) {
            headers.Authorization = `Bearer ${token}`
          }
          return headers
        },
        body: { threadId },
      }),
    [threadId],
  )

  const { messages, sendMessage, status, error } = useChat({
    id: threadId,
    messages: initialMessages,
    transport,
  })

  const streaming = status === 'streaming' || status === 'submitted'

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="border-border/60 glass-panel border-b px-6 py-4">
        <p className="font-heading text-sm font-semibold tracking-tight">
          Filing analysis
        </p>
        <p className="text-muted-foreground mt-1 text-xs">
          Grounded answers from indexed SEC documents
        </p>
      </header>

      {error && (
        <div
          className="bg-destructive/10 text-destructive flex items-center gap-2 border-b px-4 py-3 text-sm"
          role="alert"
        >
          <AlertCircle className="size-4 shrink-0" strokeWidth={2} />
          {error.message}
        </div>
      )}

      <MessageList
        messages={messages}
        loading={false}
        streaming={streaming}
      />

      <ChatInput
        value={input}
        disabled={streaming}
        onChange={setInput}
        onSubmit={() => {
          const text = input.trim()
          if (!text || streaming) {
            return
          }
          sendMessage({ text })
          setInput('')
        }}
      />
    </div>
  )
}

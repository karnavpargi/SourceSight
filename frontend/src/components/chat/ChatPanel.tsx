import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useChat } from '@ai-sdk/react'
import { DefaultChatTransport, type UIMessage } from 'ai'
import { AlertCircle } from 'lucide-react'

import { ChatInput } from '@/components/chat/ChatInput'
import { ChatSettingsMenu } from '@/components/chat/ChatSettingsMenu'
import { MessageList } from '@/components/chat/MessageList'
import { env } from '@/lib/env'
import { api } from '@/lib/api'
import {
  DEFAULT_CHAT_GENERATION,
  type ChatGenerationSettings,
} from '@/lib/chat-generation'
import type { ChatModelSelection } from '@/lib/chat-models'

interface ChatPanelProps {
  threadId: string
  initialMessages: UIMessage[]
  onThreadsChange?: () => void
}

export function ChatPanel({
  threadId,
  initialMessages,
  onThreadsChange,
}: ChatPanelProps) {
  const [input, setInput] = useState('')
  const [modelSelection, setModelSelection] = useState<ChatModelSelection | null>(
    null,
  )
  const [generationSettings, setGenerationSettings] =
    useState<ChatGenerationSettings>(DEFAULT_CHAT_GENERATION)
  const modelSelectionRef = useRef<ChatModelSelection | null>(null)
  modelSelectionRef.current = modelSelection
  const generationSettingsRef = useRef<ChatGenerationSettings>(DEFAULT_CHAT_GENERATION)
  generationSettingsRef.current = generationSettings
  const refreshedSidebarTitle = useRef(initialMessages.length > 0)

  const handleModelChange = useCallback((selection: ChatModelSelection) => {
    setModelSelection(selection)
  }, [])

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
        prepareSendMessagesRequest: ({
          body,
          id,
          messages,
          trigger,
          messageId,
        }) => {
          const selection = modelSelectionRef.current
          const generation = generationSettingsRef.current
          if (!selection) {
            throw new Error('Select a provider and model before sending.')
          }

          return {
            body: {
              ...body,
              threadId,
              provider: selection.provider,
              model: selection.model,
              temperature: generation.temperature,
              maxOutputTokens: generation.maxOutputTokens,
              id,
              messages,
              trigger,
              messageId,
            },
          }
        },
      }),
    [threadId],
  )

  const { messages, sendMessage, status, error } = useChat({
    id: threadId,
    messages: initialMessages,
    transport,
  })

  const streaming = status === 'streaming' || status === 'submitted'
  const canSend = Boolean(modelSelection) && !streaming

  useEffect(() => {
    if (refreshedSidebarTitle.current || !onThreadsChange) {
      return
    }

    if (status !== 'submitted' && status !== 'streaming') {
      return
    }

    refreshedSidebarTitle.current = true
    onThreadsChange()
  }, [onThreadsChange, status])

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
        disabled={!canSend}
        onChange={setInput}
        leadingSlot={
          <ChatSettingsMenu
            modelSelection={modelSelection}
            generationSettings={generationSettings}
            onModelChange={handleModelChange}
            onGenerationChange={setGenerationSettings}
            disabled={streaming}
          />
        }
        onSubmit={() => {
          const text = input.trim()
          if (!text || !canSend) {
            return
          }
          sendMessage({ text })
          setInput('')
        }}
      />
    </div>
  )
}

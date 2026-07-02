import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useChat } from '@ai-sdk/react'
import { DefaultChatTransport, type UIMessage } from 'ai'
import { AlertCircle, Menu } from 'lucide-react'

import { ChatInput } from '@/components/chat/ChatInput'
import { ChatSettingsMenu } from '@/components/chat/ChatSettingsMenu'
import { MessageList } from '@/components/chat/MessageList'
import { Button } from '@/components/ui/button'
import { env } from '@/lib/env'
import { api } from '@/lib/api'
import {
  DEFAULT_CHAT_GENERATION,
  type ChatGenerationSettings,
} from '@/lib/chat-generation'
import { consumePendingPrompt } from '@/lib/chat-prompts'
import type { ChatModelSelection } from '@/lib/chat-models'

interface ChatPanelProps {
  threadId: string
  threadTitle: string
  initialMessages: UIMessage[]
  onOpenSidebar?: () => void
  onThreadsChange?: () => void
}

export function ChatPanel({
  threadId,
  threadTitle,
  initialMessages,
  onOpenSidebar,
  onThreadsChange,
}: ChatPanelProps) {
  const [input, setInput] = useState('')
  const [modelSelection, setModelSelection] = useState<ChatModelSelection | null>(
    null,
  )
  const [generationSettings, setGenerationSettings] =
    useState<ChatGenerationSettings>(DEFAULT_CHAT_GENERATION)
  const refreshedSidebarTitle = useRef(initialMessages.length > 0)
  const sentInitialPrompt = useRef(false)
  const [pendingPrompt] = useState(() => consumePendingPrompt(threadId))
  const modelSelectionRef = useRef<ChatModelSelection | null>(null)
  const generationSettingsRef = useRef<ChatGenerationSettings>(DEFAULT_CHAT_GENERATION)

  const handleModelChange = useCallback((selection: ChatModelSelection) => {
    modelSelectionRef.current = selection
    setModelSelection(selection)
  }, [])

  const handleGenerationChange = useCallback((settings: ChatGenerationSettings) => {
    generationSettingsRef.current = settings
    setGenerationSettings(settings)
  }, [])

  /* eslint-disable react-hooks/refs -- prepareSendMessagesRequest reads refs at send time; useChat keeps the first transport instance */
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
  /* eslint-enable react-hooks/refs */

  const { messages, sendMessage, status, error } = useChat({
    id: threadId,
    messages: initialMessages,
    transport,
  })

  const streaming = status === 'streaming' || status === 'submitted'
  const modelReady = Boolean(modelSelection)
  const canSend = modelReady && !streaming

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

  useEffect(() => {
    if (
      !pendingPrompt ||
      sentInitialPrompt.current ||
      !modelSelection ||
      streaming ||
      messages.length > 0
    ) {
      return
    }

    sentInitialPrompt.current = true
    sendMessage({ text: pendingPrompt })
  }, [
    messages.length,
    modelSelection,
    pendingPrompt,
    sendMessage,
    streaming,
  ])

  const handlePromptSelect = useCallback(
    (prompt: string) => {
      if (!canSend) {
        return
      }
      sendMessage({ text: prompt })
    },
    [canSend, sendMessage],
  )

  const disabledHint = !modelReady
    ? 'Select a model in settings before sending.'
    : streaming
      ? 'Waiting for the current response…'
      : null

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="border-border/60 glass-panel flex items-start gap-3 border-b px-4 py-4 sm:px-6">
        {onOpenSidebar && (
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            className="mt-0.5 shrink-0 cursor-pointer md:hidden"
            aria-label="Open threads"
            onClick={onOpenSidebar}
          >
            <Menu className="size-5" strokeWidth={2} />
          </Button>
        )}
        <div className="min-w-0 flex-1">
          <p className="font-heading truncate text-sm font-semibold tracking-tight">
            {threadTitle}
          </p>
          <p className="text-muted-foreground mt-1 text-xs">
            Grounded answers from indexed SEC documents
          </p>
        </div>
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
        onSelectPrompt={handlePromptSelect}
        promptsDisabled={!canSend}
      />

      <ChatInput
        value={input}
        disabled={!canSend}
        busy={streaming}
        disabledHint={disabledHint}
        onChange={setInput}
        leadingSlot={
          <ChatSettingsMenu
            modelSelection={modelSelection}
            generationSettings={generationSettings}
            onModelChange={handleModelChange}
            onGenerationChange={handleGenerationChange}
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

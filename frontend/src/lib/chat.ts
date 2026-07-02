import type { UIMessage } from 'ai'

import type { MessageSummary } from '@/lib/api'

export function toUIMessage(message: MessageSummary): UIMessage {
  if (message.message_data) {
    return message.message_data as unknown as UIMessage
  }

  return {
    id: message.id,
    role: message.role === 'assistant' ? 'assistant' : 'user',
    parts: [{ type: 'text', text: message.content }],
  }
}

export function messageText(message: UIMessage): string {
  return message.parts
    .filter((part): part is Extract<typeof part, { type: 'text' }> => part.type === 'text')
    .map((part) => part.text)
    .join('')
}

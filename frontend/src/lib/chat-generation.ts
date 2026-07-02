export interface ChatGenerationSettings {
  temperature: number
  maxOutputTokens: number | null
}

export const DEFAULT_CHAT_GENERATION: ChatGenerationSettings = {
  temperature: 1,
  maxOutputTokens: 300,
}

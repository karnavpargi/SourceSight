import { env } from '@/lib/env'
import { ApiError, request } from '@/lib/http'
import { supabase } from '@/lib/supabase'

export interface ThreadSummary {
  id: string
  title: string
  created_at: string
  updated_at: string
}

export interface MessageSummary {
  id: string
  role: string
  content: string
  created_at: string
  message_data?: Record<string, unknown> | null
}

export interface CreateThreadRequest {
  title: string
}

export interface ChatModelOption {
  id: string
  label: string
}

export interface ChatProviderCatalog {
  id: 'local' | 'google' | 'opencode'
  label: string
  default_model: string
  models: ChatModelOption[]
}

export interface ChatProvidersResponse {
  default_provider: 'local' | 'google' | 'opencode'
  default_model: string
  providers: ChatProviderCatalog[]
}

async function getAccessToken(): Promise<string | null> {
  const { data } = await supabase.auth.getSession()
  return data.session?.access_token ?? null
}

async function authedRequest<T>(
  path: string,
  options: Omit<Parameters<typeof request>[1], 'token'> = {},
): Promise<T> {
  const token = await getAccessToken()
  if (!token) {
    throw new ApiError('Not authenticated', { status: 401 })
  }

  return request<T>(`${env.apiBaseUrl}${path}`, { ...options, token })
}

export const api = {
  get: <T>(path: string) => authedRequest<T>(path, { method: 'GET' }),

  post: <T>(path: string, body: unknown) =>
    authedRequest<T>(path, { method: 'POST', body }),

  put: <T>(path: string, body: unknown) =>
    authedRequest<T>(path, { method: 'PUT', body }),

  patch: <T>(path: string, body: unknown) =>
    authedRequest<T>(path, { method: 'PATCH', body }),

  delete: <T>(path: string) => authedRequest<T>(path, { method: 'DELETE' }),

  listThreads: () => api.get<ThreadSummary[]>('/threads'),

  createThread: (body: CreateThreadRequest) =>
    api.post<ThreadSummary>('/threads', body),

  listThreadMessages: (threadId: string) =>
    api.get<MessageSummary[]>(`/threads/${threadId}/messages`),

  listChatProviders: () => api.get<ChatProvidersResponse>('/chat/providers'),

  getAccessToken,
}

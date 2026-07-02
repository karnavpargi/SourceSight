export class ApiError extends Error {
  readonly status: number
  readonly detail: string | undefined
  readonly isNetworkError: boolean

  constructor(
    message: string,
    options: {
      status?: number
      detail?: string
      isNetworkError?: boolean
    } = {},
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = options.status ?? 0
    this.detail = options.detail
    this.isNetworkError = options.isNetworkError ?? false
  }
}

export function isNetworkError(error: unknown): boolean {
  return error instanceof ApiError && error.isNetworkError
}

type RequestOptions = Omit<RequestInit, 'body'> & {
  body?: unknown
  timeoutMs?: number
  token?: string | null
}

export async function request<T>(
  url: string,
  options: RequestOptions = {},
): Promise<T> {
  const {
    body,
    timeoutMs = 30_000,
    token,
    headers: customHeaders,
    ...init
  } = options

  const headers = new Headers(customHeaders)
  if (body !== undefined) {
    headers.set('Content-Type', 'application/json')
  }
  if (token) {
    headers.set('Authorization', `Bearer ${token}`)
  }

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const response = await fetch(url, {
      ...init,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    })

    if (!response.ok) {
      let detail: string | undefined
      try {
        const payload = (await response.json()) as { detail?: unknown }
        detail = typeof payload.detail === 'string' ? payload.detail : undefined
      } catch {
        detail = undefined
      }

      throw new ApiError(`Request failed with status ${response.status}`, {
        status: response.status,
        detail,
      })
    }

    if (response.status === 204) {
      return undefined as T
    }

    return (await response.json()) as T
  } catch (error) {
    if (error instanceof ApiError) {
      throw error
    }

    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new ApiError('Request timed out', { isNetworkError: true })
    }

    const message =
      error instanceof Error ? error.message : 'Network request failed'
    throw new ApiError(message, { isNetworkError: true })
  } finally {
    clearTimeout(timeout)
  }
}

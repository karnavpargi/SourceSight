type SessionExpiredHandler = () => void

let sessionExpiredHandler: SessionExpiredHandler | null = null

export function registerSessionExpiredHandler(handler: SessionExpiredHandler): () => void {
  sessionExpiredHandler = handler
  return () => {
    if (sessionExpiredHandler === handler) {
      sessionExpiredHandler = null
    }
  }
}

export function notifySessionExpired(): void {
  sessionExpiredHandler?.()
}

import { useEffect, useState } from 'react'

import { api } from '@/lib/api'
import {
  initialSelectionFromCatalog,
  type ChatModelSelection,
  type ChatProvidersResponse,
} from '@/lib/chat-models'

interface UseChatProviderCatalogResult {
  catalog: ChatProvidersResponse | null
  error: string | null
  loading: boolean
}

export function useChatProviderCatalog(
  onInitialSelection: (selection: ChatModelSelection) => void,
): UseChatProviderCatalogResult {
  const [catalog, setCatalog] = useState<ChatProvidersResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    async function loadCatalog() {
      try {
        const response = await api.listChatProviders()
        if (cancelled) {
          return
        }

        setCatalog(response)
        onInitialSelection(initialSelectionFromCatalog(response))
      } catch (loadError) {
        if (cancelled) {
          return
        }

        setError(
          loadError instanceof Error
            ? loadError.message
            : 'Could not load model providers',
        )
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void loadCatalog()

    return () => {
      cancelled = true
    }
  }, [onInitialSelection])

  return { catalog, error, loading }
}

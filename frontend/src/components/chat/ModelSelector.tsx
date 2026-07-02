import { useEffect, useState } from 'react'

import { api } from '@/lib/api'
import {
  initialSelectionFromCatalog,
  modelsForProvider,
  selectionForProvider,
  type ChatModelSelection,
  type ChatProvidersResponse,
  type ChatProviderId,
} from '@/lib/chat-models'

interface ModelSelectorProps {
  value: ChatModelSelection | null
  onChange: (selection: ChatModelSelection) => void
}

export function ModelSelector({ value, onChange }: ModelSelectorProps) {
  const [catalog, setCatalog] = useState<ChatProvidersResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function loadCatalog() {
      try {
        const response = await api.listChatProviders()
        if (cancelled) {
          return
        }

        setCatalog(response)
        onChange(initialSelectionFromCatalog(response))
      } catch (loadError) {
        if (cancelled) {
          return
        }

        setError(
          loadError instanceof Error
            ? loadError.message
            : 'Could not load model providers',
        )
      }
    }

    void loadCatalog()

    return () => {
      cancelled = true
    }
  }, [onChange])

  if (error) {
    return (
      <p className="text-destructive text-xs">{error}</p>
    )
  }

  if (!catalog || !value) {
    return (
      <p className="text-muted-foreground text-xs">Loading models…</p>
    )
  }

  const providerModels = modelsForProvider(catalog, value.provider)

  return (
    <div className="flex flex-wrap items-center gap-3">
      <label className="flex min-w-[10rem] flex-1 flex-col gap-1">
        <span className="text-muted-foreground text-[11px] font-medium uppercase tracking-wide">
          Provider
        </span>
        <select
          className="border-input bg-background focus-visible:ring-ring h-9 rounded-md border px-3 text-sm outline-none focus-visible:ring-2"
          value={value.provider}
          onChange={(event) => {
            const provider = event.target.value as ChatProviderId
            onChange(selectionForProvider(catalog, provider))
          }}
        >
          {catalog.providers.map((provider) => (
            <option key={provider.id} value={provider.id}>
              {provider.label}
            </option>
          ))}
        </select>
      </label>

      <label className="flex min-w-[12rem] flex-[1.5] flex-col gap-1">
        <span className="text-muted-foreground text-[11px] font-medium uppercase tracking-wide">
          Model
        </span>
        <select
          className="border-input bg-background focus-visible:ring-ring h-9 rounded-md border px-3 text-sm outline-none focus-visible:ring-2"
          value={value.model}
          onChange={(event) => {
            onChange({
              provider: value.provider,
              model: event.target.value,
            })
          }}
        >
          {providerModels.map((model) => (
            <option key={model.id} value={model.id}>
              {model.label}
            </option>
          ))}
        </select>
      </label>
    </div>
  )
}

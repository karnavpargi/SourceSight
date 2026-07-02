import { useCallback, useEffect, useId, useRef, useState } from 'react'
import { ChevronDown, Loader2, SlidersHorizontal } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useChatProviderCatalog } from '@/hooks/useChatProviderCatalog'
import {
  DEFAULT_CHAT_GENERATION,
  type ChatGenerationSettings,
} from '@/lib/chat-generation'
import {
  modelsForProvider,
  selectionForProvider,
  type ChatModelSelection,
  type ChatProviderId,
} from '@/lib/chat-models'
import { cn } from '@/lib/utils'

interface ChatSettingsMenuProps {
  modelSelection: ChatModelSelection | null
  generationSettings: ChatGenerationSettings
  onModelChange: (selection: ChatModelSelection) => void
  onGenerationChange: (settings: ChatGenerationSettings) => void
  disabled?: boolean
}

const selectClassName =
  'border-input bg-secondary/40 focus-visible:ring-ring h-9 w-full cursor-pointer rounded-md border px-3 text-sm outline-none transition-colors duration-200 focus-visible:ring-2'

export function ChatSettingsMenu({
  modelSelection,
  generationSettings,
  onModelChange,
  onGenerationChange,
  disabled = false,
}: ChatSettingsMenuProps) {
  const [open, setOpen] = useState(false)
  const menuId = useId()
  const rootRef = useRef<HTMLDivElement>(null)

  const handleInitialSelection = useCallback(
    (selection: ChatModelSelection) => {
      if (modelSelection === null) {
        onModelChange(selection)
      }
    },
    [modelSelection, onModelChange],
  )

  const { catalog, error, loading } = useChatProviderCatalog(handleInitialSelection)

  useEffect(() => {
    if (!open) {
      return
    }

    function handlePointerDown(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false)
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setOpen(false)
      }
    }

    document.addEventListener('mousedown', handlePointerDown)
    document.addEventListener('keydown', handleKeyDown)

    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [open])

  const providerModels =
    catalog && modelSelection
      ? modelsForProvider(catalog, modelSelection.provider)
      : []

  const modelLabel =
    providerModels.find((model) => model.id === modelSelection?.model)?.label ??
    modelSelection?.model ??
    'Model'

  const triggerLabel = loading
    ? 'Loading…'
    : error
      ? 'Settings'
      : modelLabel

  return (
    <div ref={rootRef} className="relative shrink-0">
      <Button
        type="button"
        variant="outline"
        disabled={disabled || loading || Boolean(error) || !modelSelection}
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-controls={menuId}
        className={cn(
          'border-glass-border bg-secondary/40 hover:bg-secondary/60 h-11 max-w-[2.75rem] cursor-pointer gap-1.5 px-2.5 transition-colors duration-200 sm:max-w-[12rem] sm:px-3',
          open && 'bg-secondary/60 ring-ring/50 ring-[3px]',
        )}
        onClick={() => setOpen((current) => !current)}
      >
        {loading ? (
          <Loader2 className="size-4 animate-spin" strokeWidth={2} />
        ) : (
          <SlidersHorizontal className="size-4 shrink-0" strokeWidth={2} />
        )}
        <span className="hidden min-w-0 truncate sm:inline">{triggerLabel}</span>
        <ChevronDown
          className={cn(
            'hidden size-3.5 shrink-0 opacity-60 transition-transform duration-200 sm:inline',
            open && 'rotate-180',
          )}
          strokeWidth={2}
        />
      </Button>

      {open && (
        <div
          id={menuId}
          role="dialog"
          aria-label="Chat model settings"
          className="border-glass-border bg-popover absolute bottom-full left-0 z-50 mb-2 w-[min(100vw-2rem,20rem)] rounded-2xl border p-4 shadow-2xl backdrop-blur-xl"
        >
          <div className="mb-3">
            <p className="text-sm font-medium">Model settings</p>
            <p className="text-muted-foreground mt-0.5 text-xs">
              Applies to your next message
            </p>
          </div>

          {error ? (
            <p className="text-destructive text-xs">{error}</p>
          ) : !catalog || !modelSelection ? (
            <p className="text-muted-foreground text-xs">Loading models…</p>
          ) : (
            <div className="space-y-3">
              <label className="flex flex-col gap-1.5">
                <span className="text-muted-foreground text-xs font-medium">
                  Provider
                </span>
                <select
                  className={selectClassName}
                  value={modelSelection.provider}
                  onChange={(event) => {
                    const provider = event.target.value as ChatProviderId
                    onModelChange(selectionForProvider(catalog, provider))
                  }}
                >
                  {catalog.providers.map((provider) => (
                    <option key={provider.id} value={provider.id}>
                      {provider.label}
                    </option>
                  ))}
                </select>
              </label>

              <label className="flex flex-col gap-1.5">
                <span className="text-muted-foreground text-xs font-medium">
                  Model
                </span>
                <select
                  className={selectClassName}
                  value={modelSelection.model}
                  onChange={(event) => {
                    onModelChange({
                      provider: modelSelection.provider,
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

              <label className="flex flex-col gap-1.5">
                <span className="text-muted-foreground text-xs font-medium">
                  Temperature ({generationSettings.temperature.toFixed(1)})
                </span>
                <input
                  type="range"
                  min={0}
                  max={2}
                  step={0.1}
                  value={generationSettings.temperature}
                  className="accent-primary h-2 w-full cursor-pointer"
                  onChange={(event) => {
                    onGenerationChange({
                      ...generationSettings,
                      temperature: Number.parseFloat(event.target.value),
                    })
                  }}
                />
              </label>

              <label className="flex flex-col gap-1.5">
                <span className="text-muted-foreground text-xs font-medium">
                  Max response tokens
                </span>
                <Input
                  type="number"
                  min={1}
                  placeholder={String(DEFAULT_CHAT_GENERATION.maxOutputTokens)}
                  value={generationSettings.maxOutputTokens ?? ''}
                  className="bg-secondary/40 border-glass-border h-9"
                  onChange={(event) => {
                    const raw = event.target.value.trim()
                    onGenerationChange({
                      ...generationSettings,
                      maxOutputTokens:
                        raw === '' ? null : Number.parseInt(raw, 10),
                    })
                  }}
                />
                <span className="text-muted-foreground text-[11px]">
                  Limits answer text only.{' '}
                  {generationSettings.maxOutputTokens === null
                    ? 'Unlimited.'
                    : `≈ ${Math.max(1, Math.round(generationSettings.maxOutputTokens * 0.75))} words.`}
                </span>
              </label>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

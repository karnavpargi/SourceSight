import { useCallback } from 'react'
import { ChevronDown, Loader2, SlidersHorizontal } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Popover,
  PopoverContent,
  PopoverDescription,
  PopoverHeader,
  PopoverTitle,
  PopoverTrigger,
} from '@/components/ui/popover'
import { useChatProviderCatalog } from '@/hooks/useChatProviderCatalog'
import {
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
  const handleInitialSelection = useCallback(
    (selection: ChatModelSelection) => {
      if (modelSelection === null) {
        onModelChange(selection)
      }
    },
    [modelSelection, onModelChange],
  )

  const { catalog, error, loading } = useChatProviderCatalog(handleInitialSelection)

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
    <Popover>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="outline"
          disabled={disabled || loading || Boolean(error) || !modelSelection}
          aria-label={`Model settings: ${triggerLabel}`}
          className={cn(
            'border-glass-border bg-secondary/40 hover:bg-secondary/60 h-11 max-w-[2.75rem] cursor-pointer gap-1.5 px-2.5 transition-colors duration-200 sm:max-w-[12rem] sm:px-3',
          )}
        >
          {loading ? (
            <Loader2 className="size-4 animate-spin" strokeWidth={2} />
          ) : (
            <SlidersHorizontal className="size-4 shrink-0" strokeWidth={2} />
          )}
          <span className="hidden min-w-0 truncate sm:inline">{triggerLabel}</span>
          <ChevronDown
            className="hidden size-3.5 shrink-0 opacity-60 sm:inline"
            strokeWidth={2}
          />
        </Button>
      </PopoverTrigger>

      <PopoverContent
        side="top"
        align="start"
        sideOffset={8}
        className="border-glass-border w-[min(100vw-2rem,20rem)] rounded-2xl p-4 backdrop-blur-xl"
      >
        <PopoverHeader className="mb-3">
          <PopoverTitle>Model settings</PopoverTitle>
          <PopoverDescription className="text-xs">
            Applies to your next message
          </PopoverDescription>
        </PopoverHeader>

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
                aria-valuemin={0}
                aria-valuemax={2}
                aria-valuenow={generationSettings.temperature}
                className="accent-primary h-2 w-full cursor-pointer"
                onChange={(event) => {
                  onGenerationChange({
                    ...generationSettings,
                    temperature: Number.parseFloat(event.target.value),
                  })
                }}
              />
            </label>
          </div>
        )}
      </PopoverContent>
    </Popover>
  )
}

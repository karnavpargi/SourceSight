import { SlidersHorizontal } from 'lucide-react'

import { Button } from '@/components/ui/button'
import {
  Popover,
  PopoverContent,
  PopoverDescription,
  PopoverHeader,
  PopoverTitle,
  PopoverTrigger,
} from '@/components/ui/popover'
import {
  type ChatGenerationSettings,
} from '@/lib/chat-generation'
import { cn } from '@/lib/utils'

interface ChatSettingsMenuProps {
  generationSettings: ChatGenerationSettings
  onGenerationChange: (settings: ChatGenerationSettings) => void
  disabled?: boolean
}

export function ChatSettingsMenu({
  generationSettings,
  onGenerationChange,
  disabled = false,
}: ChatSettingsMenuProps) {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="outline"
          disabled={disabled}
          aria-label="Chat settings"
          className={cn(
            'border-glass-border bg-secondary/40 hover:bg-secondary/60 h-11 max-w-[2.75rem] cursor-pointer gap-1.5 px-2.5 transition-colors duration-200 sm:max-w-[12rem] sm:px-3',
          )}
        >
          <SlidersHorizontal className="size-4 shrink-0" strokeWidth={2} />
          <span className="hidden min-w-0 truncate sm:inline">Settings</span>
        </Button>
      </PopoverTrigger>

      <PopoverContent
        side="top"
        align="start"
        sideOffset={8}
        className="border-glass-border w-[min(100vw-2rem,20rem)] rounded-2xl p-4 backdrop-blur-xl"
      >
        <PopoverHeader className="mb-3">
          <PopoverTitle>Chat settings</PopoverTitle>
          <PopoverDescription className="text-xs">
            Model is configured on the server. Temperature applies to your next
            message.
          </PopoverDescription>
        </PopoverHeader>

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
      </PopoverContent>
    </Popover>
  )
}

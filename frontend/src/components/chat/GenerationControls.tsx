import { Input } from '@/components/ui/input'
import type { ChatGenerationSettings } from '@/lib/chat-generation'

interface GenerationControlsProps {
  value: ChatGenerationSettings
  onChange: (value: ChatGenerationSettings) => void
}

export function GenerationControls({ value, onChange }: GenerationControlsProps) {
  return (
    <div className="flex flex-wrap items-end gap-3">
      <label className="flex min-w-[10rem] flex-1 flex-col gap-1">
        <span className="text-muted-foreground text-[11px] font-medium uppercase tracking-wide">
          Temperature ({value.temperature.toFixed(1)})
        </span>
        <input
          type="range"
          min={0}
          max={2}
          step={0.1}
          value={value.temperature}
          className="accent-primary h-2 w-full cursor-pointer"
          onChange={(event) => {
            onChange({
              ...value,
              temperature: Number.parseFloat(event.target.value),
            })
          }}
        />
      </label>

      <label className="flex min-w-[10rem] flex-1 flex-col gap-1">
        <span className="text-muted-foreground text-[11px] font-medium uppercase tracking-wide">
          Max response tokens
        </span>
        <Input
          type="number"
          min={1}
          placeholder="300"
          value={value.maxOutputTokens ?? ''}
          className="h-9"
          onChange={(event) => {
            const raw = event.target.value.trim()
            onChange({
              ...value,
              maxOutputTokens: raw === '' ? null : Number.parseInt(raw, 10),
            })
          }}
        />
      </label>
    </div>
  )
}

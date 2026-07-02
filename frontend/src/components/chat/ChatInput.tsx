import { Loader2, SendHorizontal } from 'lucide-react'
import type { ReactNode } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

interface ChatInputProps {
  value: string
  disabled: boolean
  onChange: (value: string) => void
  onSubmit: () => void
  leadingSlot?: ReactNode
}

export function ChatInput({
  value,
  disabled,
  onChange,
  onSubmit,
  leadingSlot,
}: ChatInputProps) {
  const canSubmit = !disabled && value.trim().length > 0

  return (
    <div className="border-border/60 bg-background/80 border-t p-4 backdrop-blur-xl">
      <form
        className="mx-auto flex w-full max-w-3xl items-end gap-2"
        onSubmit={(event) => {
          event.preventDefault()
          onSubmit()
        }}
      >
        {leadingSlot}
        <Input
          value={value}
          disabled={disabled}
          placeholder="Ask about a filing, metric, or risk factor…"
          className="bg-secondary/40 border-glass-border h-11 flex-1"
          onChange={(event) => onChange(event.target.value)}
        />
        <Button
          type="submit"
          disabled={!canSubmit}
          className="bg-cta text-cta-foreground hover:bg-cta/90 h-11 min-w-11 cursor-pointer px-4 transition-colors duration-200 disabled:cursor-not-allowed"
        >
          {disabled ? (
            <Loader2 className="size-4 animate-spin" strokeWidth={2} />
          ) : (
            <>
              <SendHorizontal className="size-4 sm:hidden" strokeWidth={2} />
              <span className="hidden sm:inline">Send</span>
            </>
          )}
        </Button>
      </form>
    </div>
  )
}

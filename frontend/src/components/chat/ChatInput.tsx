import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from 'react'
import { Loader2, SendHorizontal } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  filterExamplePrompts,
  type ExamplePrompt,
} from '@/lib/chat-prompts'
import { cn } from '@/lib/utils'

interface ChatInputProps {
  value: string
  disabled: boolean
  busy?: boolean
  disabledHint?: string | null
  onChange: (value: string) => void
  onSubmit: () => void
  leadingSlot?: ReactNode
}

export function ChatInput({
  value,
  disabled,
  busy = false,
  disabledHint = null,
  onChange,
  onSubmit,
  leadingSlot,
}: ChatInputProps) {
  const listboxId = useId()
  const rootRef = useRef<HTMLDivElement>(null)
  const [suggestionsOpen, setSuggestionsOpen] = useState(false)
  const [activeIndex, setActiveIndex] = useState(-1)

  const canSubmit = !disabled && !busy && value.trim().length > 0
  const inputDisabled = disabled || busy

  const suggestions = useMemo(
    () => (suggestionsOpen ? filterExamplePrompts(value) : []),
    [suggestionsOpen, value],
  )

  const showSuggestions = suggestionsOpen && suggestions.length > 0 && !inputDisabled
  const highlightedIndex =
    activeIndex >= 0 && activeIndex < suggestions.length ? activeIndex : -1

  const selectSuggestion = useCallback(
    (item: ExamplePrompt) => {
      onChange(item.prompt)
      setSuggestionsOpen(false)
      setActiveIndex(-1)
    },
    [onChange],
  )

  useEffect(() => {
    if (!suggestionsOpen) {
      return
    }

    function handlePointerDown(event: MouseEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setSuggestionsOpen(false)
        setActiveIndex(-1)
      }
    }

    document.addEventListener('mousedown', handlePointerDown)

    return () => {
      document.removeEventListener('mousedown', handlePointerDown)
    }
  }, [suggestionsOpen])

  function handleInputKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (!showSuggestions) {
      return
    }

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault()
        setActiveIndex((current) =>
          current >= suggestions.length - 1 ? 0 : current + 1,
        )
        break
      case 'ArrowUp':
        event.preventDefault()
        setActiveIndex((current) =>
          current <= 0 ? suggestions.length - 1 : current - 1,
        )
        break
      case 'Enter':
        if (highlightedIndex >= 0 && suggestions[highlightedIndex]) {
          event.preventDefault()
          selectSuggestion(suggestions[highlightedIndex])
        }
        break
      case 'Escape':
        event.preventDefault()
        setSuggestionsOpen(false)
        setActiveIndex(-1)
        break
      case 'Tab':
        setSuggestionsOpen(false)
        setActiveIndex(-1)
        break
      default:
        break
    }
  }

  return (
    <div className="border-border/60 bg-background/80 border-t p-4 backdrop-blur-xl">
      <form
        className="mx-auto flex w-full max-w-3xl flex-col gap-2"
        onSubmit={(event) => {
          event.preventDefault()
          onSubmit()
        }}
      >
        <div className="flex items-end gap-2">
          {leadingSlot}
          <div ref={rootRef} className="relative min-w-0 flex-1">
            <label htmlFor="chat-message" className="sr-only">
              Message
            </label>
            <Input
              id="chat-message"
              value={value}
              disabled={inputDisabled}
              placeholder="Ask about revenue mix, AWS margins, risk factors…"
              role="combobox"
              aria-expanded={showSuggestions}
              aria-controls={showSuggestions ? listboxId : undefined}
              aria-autocomplete="list"
              aria-activedescendant={
                showSuggestions && highlightedIndex >= 0
                  ? `${listboxId}-option-${highlightedIndex}`
                  : undefined
              }
              className="bg-secondary/40 border-glass-border h-11 w-full"
              onFocus={() => {
                if (!inputDisabled) {
                  setSuggestionsOpen(true)
                }
              }}
              onChange={(event) => {
                onChange(event.target.value)
                setSuggestionsOpen(true)
                setActiveIndex(-1)
              }}
              onKeyDown={handleInputKeyDown}
            />

            {showSuggestions && (
              <ul
                id={listboxId}
                role="listbox"
                aria-label="Example analyst questions"
                className="border-glass-border bg-popover absolute bottom-full left-0 z-50 mb-2 max-h-56 w-full overflow-y-auto rounded-2xl border p-1 shadow-2xl backdrop-blur-xl"
              >
                {suggestions.map((item, index) => (
                  <li key={item.id} role="presentation">
                    <button
                      type="button"
                      id={`${listboxId}-option-${index}`}
                      role="option"
                      aria-selected={index === highlightedIndex}
                      className={cn(
                        'hover:bg-accent/50 w-full cursor-pointer rounded-xl px-3 py-2 text-left transition-colors duration-200',
                        index === highlightedIndex && 'bg-accent/50',
                      )}
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={() => selectSuggestion(item)}
                    >
                      <p className="text-foreground truncate text-sm font-medium">
                        {item.label}
                      </p>
                      <p className="text-muted-foreground line-clamp-1 text-xs">
                        {item.prompt}
                      </p>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <Button
            type="submit"
            disabled={!canSubmit}
            aria-busy={busy}
            className="bg-cta text-cta-foreground hover:bg-cta/90 h-11 min-w-11 shrink-0 cursor-pointer px-4 transition-colors duration-200 disabled:cursor-not-allowed"
          >
            {busy ? (
              <Loader2 className="size-4 animate-spin" strokeWidth={2} />
            ) : (
              <>
                <SendHorizontal className="size-4 sm:hidden" strokeWidth={2} />
                <span className="hidden sm:inline">Send</span>
              </>
            )}
          </Button>
        </div>

        {disabledHint && (
          <p className="text-muted-foreground text-xs" role="status">
            {disabledHint}
          </p>
        )}
      </form>
    </div>
  )
}

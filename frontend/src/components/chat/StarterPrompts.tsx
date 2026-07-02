import { EXAMPLE_PROMPTS } from '@/lib/chat-prompts'
import { cn } from '@/lib/utils'

interface StarterPromptsProps {
  onSelect: (prompt: string) => void
  disabled?: boolean
  className?: string
}

export function StarterPrompts({
  onSelect,
  disabled = false,
  className,
}: StarterPromptsProps) {
  return (
    <div className={cn('space-y-3', className)}>
      <p className="text-muted-foreground text-center text-xs font-medium tracking-wide uppercase">
        Example analyst questions
      </p>
      <ul className="max-h-[min(24rem,50vh)] space-y-2 overflow-y-auto pr-1">
        {EXAMPLE_PROMPTS.map((item) => (
          <li key={item.id}>
            <button
              type="button"
              disabled={disabled}
              title={item.prompt}
              className="glass-panel hover:border-primary/30 hover:bg-primary/5 w-full cursor-pointer rounded-xl px-4 py-3 text-left transition-colors duration-200 disabled:cursor-not-allowed disabled:opacity-50"
              onClick={() => onSelect(item.prompt)}
            >
              <p className="text-foreground text-sm font-medium">{item.label}</p>
              <p className="text-muted-foreground mt-1 line-clamp-2 text-xs leading-relaxed">
                {item.prompt}
              </p>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}

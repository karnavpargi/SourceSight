import type { CSSProperties } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  Info,
  Loader2,
  XCircle,
} from 'lucide-react'
import { Toaster as Sonner, type ToasterProps } from 'sonner'

const Toaster = ({ ...props }: ToasterProps) => {
  return (
    <Sonner
      theme="system"
      className="toaster group"
      icons={{
        success: <CheckCircle2 className="size-4" strokeWidth={2} />,
        info: <Info className="size-4" strokeWidth={2} />,
        warning: <AlertTriangle className="size-4" strokeWidth={2} />,
        error: <XCircle className="size-4" strokeWidth={2} />,
        loading: <Loader2 className="size-4 animate-spin" strokeWidth={2} />,
      }}
      style={
        {
          '--normal-bg': 'var(--popover)',
          '--normal-text': 'var(--popover-foreground)',
          '--normal-border': 'var(--border)',
          '--border-radius': 'var(--radius)',
        } as CSSProperties
      }
      toastOptions={{
        classNames: {
          toast: 'cn-toast',
        },
      }}
      {...props}
    />
  )
}

export { Toaster }

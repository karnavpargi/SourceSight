import { Component, type ErrorInfo, type ReactNode } from 'react'
import { AlertTriangle, Copy } from 'lucide-react'

import { Button } from '@/components/ui/button'

interface ErrorBoundaryProps {
  children: ReactNode
}

interface ErrorBoundaryState {
  error: Error | null
  errorId: string | null
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = {
    error: null,
    errorId: null,
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      error,
      errorId: crypto.randomUUID(),
    }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('Unhandled UI error', error, info.componentStack)
  }

  private handleReload = (): void => {
    window.location.reload()
  }

  private handleCopyReport = async (): Promise<void> => {
    const { error, errorId } = this.state
    const report = [
      `SourceSight error report`,
      `id: ${errorId ?? 'unknown'}`,
      `message: ${error?.message ?? 'unknown'}`,
      `url: ${window.location.href}`,
      `time: ${new Date().toISOString()}`,
    ].join('\n')

    try {
      await navigator.clipboard.writeText(report)
    } catch {
      console.error(report)
    }
  }

  render(): ReactNode {
    const { error, errorId } = this.state

    if (!error) {
      return this.props.children
    }

    return (
      <div className="flex min-h-svh items-center justify-center p-6">
        <div className="glass-panel max-w-lg rounded-3xl p-8 text-center">
          <div className="bg-destructive/10 text-destructive mx-auto mb-4 flex size-12 items-center justify-center rounded-2xl">
            <AlertTriangle className="size-6" strokeWidth={1.75} />
          </div>
          <h1 className="font-heading text-xl font-semibold tracking-tight">
            Something went wrong
          </h1>
          <p className="text-muted-foreground mt-2 text-sm leading-relaxed">
            The interface hit an unexpected error. You can reload the page or copy a
            short report for debugging.
          </p>
          {errorId && (
            <p className="text-muted-foreground mt-3 font-mono text-xs">
              Report ID: {errorId}
            </p>
          )}
          <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
            <Button
              type="button"
              className="cursor-pointer"
              onClick={() => void this.handleCopyReport()}
            >
              <Copy className="size-4" strokeWidth={2} />
              Copy report
            </Button>
            <Button
              type="button"
              variant="outline"
              className="cursor-pointer"
              onClick={this.handleReload}
            >
              Reload page
            </Button>
          </div>
        </div>
      </div>
    )
  }
}

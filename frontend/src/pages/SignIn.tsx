import {
  FileSearch,
  Lock,
  Mail,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { supabase } from '@/lib/supabase'

const TRUST_POINTS = [
  {
    icon: ShieldCheck,
    title: 'Grounded answers',
    description: 'Every claim tied to indexed SEC filings.',
  },
  {
    icon: Lock,
    title: 'Secure access',
    description: 'Email-only auth for Sourceline analysts.',
  },
  {
    icon: FileSearch,
    title: 'Citation-first',
    description: 'Skip intake work and jump to insight.',
  },
] as const

export function SignIn() {
  const [email, setEmail] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [sent, setSent] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const trimmed = email.trim()
    if (!trimmed) {
      return
    }

    setSubmitting(true)
    setError(null)

    const { error: signInError } = await supabase.auth.signInWithOtp({
      email: trimmed,
      options: {
        emailRedirectTo: window.location.origin,
      },
    })

    setSubmitting(false)

    if (signInError) {
      setError(signInError.message)
      return
    }

    setSent(true)
  }

  return (
    <div className="grid min-h-svh lg:grid-cols-2">
      <section className="relative hidden overflow-hidden lg:flex lg:flex-col lg:justify-between lg:p-10">
        <div className="glass-panel absolute inset-4 rounded-3xl" aria-hidden="true" />
        <div className="relative z-10 flex flex-col gap-8 p-6">
          <div className="flex items-center gap-3">
            <div className="bg-primary/15 text-primary flex size-10 items-center justify-center rounded-xl">
              <Sparkles className="size-5" strokeWidth={2} />
            </div>
            <div>
              <p className="font-heading text-lg font-semibold tracking-tight">
                SourceSight
              </p>
              <p className="text-muted-foreground text-sm">
                Sourceline Capital · Research copilot
              </p>
            </div>
          </div>

          <div className="max-w-md space-y-4">
            <h1 className="font-heading brand-glow text-4xl leading-tight font-semibold tracking-tight">
              Document intake, answered with sources.
            </h1>
            <p className="text-muted-foreground text-base leading-relaxed">
              Query 10-Ks and 10-Qs in plain English. SourceSight returns
              citable passages so analysts can move straight to original analysis.
            </p>
          </div>

          <ul className="max-w-md space-y-4">
            {TRUST_POINTS.map(({ icon: Icon, title, description }) => (
              <li
                key={title}
                className="glass-panel flex cursor-default gap-3 rounded-2xl p-4 transition-colors duration-200"
              >
                <div className="bg-primary/10 text-primary flex size-9 shrink-0 items-center justify-center rounded-lg">
                  <Icon className="size-4" strokeWidth={2} />
                </div>
                <div>
                  <p className="text-sm font-medium">{title}</p>
                  <p className="text-muted-foreground mt-1 text-sm">{description}</p>
                </div>
              </li>
            ))}
          </ul>
        </div>

        <p className="text-muted-foreground relative z-10 px-6 pb-2 text-xs">
          Internal tool · Not investment advice
        </p>
      </section>

      <section className="flex items-center justify-center p-6 sm:p-10">
        <div className="glass-panel w-full max-w-md rounded-3xl p-6 sm:p-8">
          <div className="mb-8 space-y-2 lg:hidden">
            <div className="flex items-center gap-2">
              <Sparkles className="text-primary size-5" strokeWidth={2} />
              <p className="font-heading font-semibold">SourceSight</p>
            </div>
            <p className="text-muted-foreground text-sm">
              Sign in to access the research copilot.
            </p>
          </div>

          <div className="mb-6 hidden items-center gap-2 text-sm lg:flex">
            <Lock className="text-primary size-4" strokeWidth={2} />
            <span className="text-muted-foreground">Secure analyst sign-in</span>
          </div>

          <h2 className="font-heading mb-2 text-2xl font-semibold tracking-tight">
            {sent ? 'Check your inbox' : 'Sign in'}
          </h2>
          <p className="text-muted-foreground mb-6 text-sm">
            {sent
              ? 'Open the magic link on this device to continue.'
              : 'We’ll email you a one-time sign-in link.'}
          </p>

          {sent ? (
            <div className="space-y-4">
              <div
                className="bg-secondary/60 flex items-start gap-3 rounded-2xl p-4"
                role="status"
              >
                <Mail className="text-primary mt-0.5 size-4 shrink-0" strokeWidth={2} />
                <p className="text-sm leading-relaxed">
                  Link sent to{' '}
                  <span className="text-foreground font-medium">{email}</span>
                </p>
              </div>
              <Button
                variant="outline"
                className="w-full cursor-pointer transition-colors duration-200"
                onClick={() => {
                  setSent(false)
                  setEmail('')
                }}
              >
                Use a different email
              </Button>
            </div>
          ) : (
            <form className="space-y-4" onSubmit={(event) => void handleSubmit(event)}>
              <div className="space-y-2">
                <label htmlFor="email" className="text-sm font-medium">
                  Work email
                </label>
                <Input
                  id="email"
                  type="email"
                  autoComplete="email"
                  placeholder="you@sourceline.com"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  required
                />
              </div>

              {error && (
                <p className="text-destructive text-sm" role="alert">
                  {error}
                </p>
              )}

              <Button
                type="submit"
                disabled={submitting}
                className="bg-brand-purple hover:bg-brand-purple/90 w-full cursor-pointer text-white transition-colors duration-200"
              >
                {submitting ? 'Sending link…' : 'Send magic link'}
              </Button>
            </form>
          )}
        </div>
      </section>
    </div>
  )
}

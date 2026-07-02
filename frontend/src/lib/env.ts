type EnvKey =
  | 'VITE_API_BASE_URL'
  | 'VITE_SUPABASE_URL'
  | 'VITE_SUPABASE_ANON_KEY'

function readEnv(name: EnvKey): string {
  const value = import.meta.env[name]
  if (!value) {
    throw new Error(`Missing required env var: ${name}`)
  }
  return value
}

function readOptionalEnv(name: string): string | undefined {
  const value = import.meta.env[name]
  if (typeof value !== 'string') {
    return undefined
  }
  const trimmed = value.trim()
  return trimmed === '' ? undefined : trimmed
}

export const env = {
  apiBaseUrl: readEnv('VITE_API_BASE_URL'),
  supabaseUrl: readEnv('VITE_SUPABASE_URL'),
  supabaseAnonKey: readEnv('VITE_SUPABASE_ANON_KEY'),
  /** Pre-fills the sign-in form in `pnpm dev` only. Set `VITE_DEV_SIGN_IN_EMAIL` locally. */
  devSignInEmail: import.meta.env.DEV
    ? readOptionalEnv('VITE_DEV_SIGN_IN_EMAIL')
    : undefined,
} as const

import { Navigate, Route, Routes } from 'react-router-dom'

import { ErrorBoundary } from '@/components/ErrorBoundary'
import { Skeleton } from '@/components/ui/skeleton'
import { Toaster } from '@/components/ui/sonner'
import { useAuth } from '@/lib/auth'
import { Chat } from '@/pages/Chat'
import { SignIn } from '@/pages/SignIn'

function RequireAuth({ children }: { children: React.ReactNode }) {
  const { session, loading } = useAuth()

  if (loading) {
    return (
      <div className="flex min-h-svh items-center justify-center p-6">
        <Skeleton className="h-10 w-48" />
      </div>
    )
  }

  if (!session) {
    return <Navigate to="/sign-in" replace />
  }

  return children
}

function PublicOnly({ children }: { children: React.ReactNode }) {
  const { session, loading } = useAuth()

  if (loading) {
    return (
      <div className="flex min-h-svh items-center justify-center p-6">
        <Skeleton className="h-10 w-48" />
      </div>
    )
  }

  if (session) {
    return <Navigate to="/chat" replace />
  }

  return children
}

export default function App() {
  return (
    <ErrorBoundary>
      <Routes>
        <Route
          path="/sign-in"
          element={
            <PublicOnly>
              <SignIn />
            </PublicOnly>
          }
        />
        <Route
          path="/chat/:threadId?"
          element={
            <RequireAuth>
              <Chat />
            </RequireAuth>
          }
        />
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
      <Toaster richColors closeButton />
    </ErrorBoundary>
  )
}

import { useCallback, useEffect, useRef } from 'react'

const STICK_THRESHOLD_PX = 80

export function useStickToBottom(scrollKey: unknown) {
  const containerRef = useRef<HTMLDivElement>(null)
  const stuckToBottomRef = useRef(true)

  const scrollToBottom = useCallback(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    container.scrollTop = container.scrollHeight
  }, [])

  const handleScroll = useCallback(() => {
    const container = containerRef.current
    if (!container) {
      return
    }

    const distanceFromBottom =
      container.scrollHeight - container.scrollTop - container.clientHeight
    stuckToBottomRef.current = distanceFromBottom <= STICK_THRESHOLD_PX
  }, [])

  useEffect(() => {
    if (!stuckToBottomRef.current) {
      return
    }

    scrollToBottom()
  }, [scrollKey, scrollToBottom])

  return { containerRef, handleScroll, scrollToBottom }
}

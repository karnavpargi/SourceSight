import type { ReactNode } from 'react'

interface MarkdownContentProps {
  content: string
}

type Block =
  | { type: 'paragraph'; text: string }
  | { type: 'heading'; level: number; text: string }
  | { type: 'ul'; items: string[] }
  | { type: 'ol'; items: string[] }
  | { type: 'code'; language: string; text: string }

function parseBlocks(content: string): Block[] {
  const blocks: Block[] = []
  const segments = content.split(/(```[\s\S]*?```)/g)

  for (const segment of segments) {
    if (segment.startsWith('```')) {
      const fence = segment.slice(3)
      const closing = fence.indexOf('\n')
      const language = closing === -1 ? '' : fence.slice(0, closing).trim()
      const text =
        closing === -1
          ? fence.replace(/```$/, '')
          : fence.slice(closing + 1).replace(/\n?```$/, '')
      blocks.push({ type: 'code', language, text: text.trimEnd() })
      continue
    }

    parseTextSegment(segment, blocks)
  }

  return blocks
}

function parseTextSegment(segment: string, blocks: Block[]): void {
  const lines = segment.replace(/\r\n/g, '\n').split('\n')
  let index = 0

  while (index < lines.length) {
    const line = lines[index]
    const trimmed = line.trim()

    if (trimmed.length === 0) {
      index += 1
      continue
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.+)$/)
    if (headingMatch) {
      blocks.push({
        type: 'heading',
        level: headingMatch[1].length,
        text: headingMatch[2],
      })
      index += 1
      continue
    }

    if (/^[-*]\s+/.test(trimmed)) {
      const items: string[] = []
      while (index < lines.length && /^[-*]\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^[-*]\s+/, ''))
        index += 1
      }
      blocks.push({ type: 'ul', items })
      continue
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      const items: string[] = []
      while (index < lines.length && /^\d+\.\s+/.test(lines[index].trim())) {
        items.push(lines[index].trim().replace(/^\d+\.\s+/, ''))
        index += 1
      }
      blocks.push({ type: 'ol', items })
      continue
    }

    const paragraphLines: string[] = [trimmed]
    index += 1
    while (
      index < lines.length &&
      lines[index].trim().length > 0 &&
      !/^#{1,3}\s+/.test(lines[index].trim()) &&
      !/^[-*]\s+/.test(lines[index].trim()) &&
      !/^\d+\.\s+/.test(lines[index].trim())
    ) {
      paragraphLines.push(lines[index].trim())
      index += 1
    }

    blocks.push({ type: 'paragraph', text: paragraphLines.join(' ') })
  }
}

function renderInline(text: string, keyPrefix: string): ReactNode[] {
  return tokenizeInline(text).map((token, index) => {
    const key = `${keyPrefix}-${index}`

    switch (token.type) {
      case 'text':
        return token.value
      case 'bold':
        return <strong key={key}>{renderInline(token.value, `${key}-b`)}</strong>
      case 'italic':
        return <em key={key}>{renderInline(token.value, `${key}-i`)}</em>
      case 'code':
        return (
          <code
            key={key}
            className="bg-secondary/60 rounded px-1 py-0.5 font-mono text-[0.85em]"
          >
            {token.value}
          </code>
        )
      case 'link':
        return (
          <a
            key={key}
            href={token.href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary underline underline-offset-2 hover:text-primary/80"
          >
            {token.label}
          </a>
        )
      default: {
        const _exhaustive: never = token
        return _exhaustive
      }
    }
  })
}

type InlineToken =
  | { type: 'text'; value: string }
  | { type: 'bold'; value: string }
  | { type: 'italic'; value: string }
  | { type: 'code'; value: string }
  | { type: 'link'; label: string; href: string }

function tokenizeInline(text: string): InlineToken[] {
  const tokens: InlineToken[] = []
  let cursor = 0

  while (cursor < text.length) {
    const slice = text.slice(cursor)
    const linkMatch = slice.match(/^\[([^\]]+)\]\(([^)]+)\)/)
    const codeMatch = slice.match(/^`([^`]+)`/)
    const boldMatch = slice.match(/^\*\*([^*]+)\*\*/)
    const italicMatch = slice.match(/^(?<!\*)\*([^*]+)\*(?!\*)/)

    if (linkMatch) {
      tokens.push({
        type: 'link',
        label: linkMatch[1],
        href: linkMatch[2],
      })
      cursor += linkMatch[0].length
      continue
    }

    if (codeMatch) {
      tokens.push({ type: 'code', value: codeMatch[1] })
      cursor += codeMatch[0].length
      continue
    }

    if (boldMatch) {
      tokens.push({ type: 'bold', value: boldMatch[1] })
      cursor += boldMatch[0].length
      continue
    }

    if (italicMatch) {
      tokens.push({ type: 'italic', value: italicMatch[1] })
      cursor += italicMatch[0].length
      continue
    }

    const nextSpecial = slice.slice(1).search(/[[*`]/)
    const end =
      nextSpecial === -1 ? text.length : cursor + 1 + nextSpecial
    tokens.push({ type: 'text', value: text.slice(cursor, end) })
    cursor = end
  }

  return mergeTextTokens(tokens)
}

function mergeTextTokens(tokens: InlineToken[]): InlineToken[] {
  const merged: InlineToken[] = []

  for (const token of tokens) {
    const previous = merged[merged.length - 1]
    if (token.type === 'text' && previous?.type === 'text') {
      previous.value += token.value
      continue
    }
    merged.push(token)
  }

  return merged
}

export function MarkdownContent({ content }: MarkdownContentProps) {
  const blocks = parseBlocks(content)

  return (
    <div className="space-y-3 break-words">
      {blocks.map((block, index) => {
        const key = `block-${index}`

        switch (block.type) {
          case 'heading': {
            const className =
              block.level === 1
                ? 'text-base font-semibold'
                : block.level === 2
                  ? 'text-sm font-semibold'
                  : 'text-sm font-medium'
            return (
              <p key={key} className={className}>
                {renderInline(block.text, key)}
              </p>
            )
          }
          case 'paragraph':
            return (
              <p key={key} className="whitespace-pre-wrap">
                {renderInline(block.text, key)}
              </p>
            )
          case 'ul':
            return (
              <ul key={key} className="list-disc space-y-1 pl-5">
                {block.items.map((item, itemIndex) => (
                  <li key={`${key}-${itemIndex}`}>
                    {renderInline(item, `${key}-${itemIndex}`)}
                  </li>
                ))}
              </ul>
            )
          case 'ol':
            return (
              <ol key={key} className="list-decimal space-y-1 pl-5">
                {block.items.map((item, itemIndex) => (
                  <li key={`${key}-${itemIndex}`}>
                    {renderInline(item, `${key}-${itemIndex}`)}
                  </li>
                ))}
              </ol>
            )
          case 'code':
            return (
              <pre
                key={key}
                className="bg-secondary/60 overflow-x-auto rounded-lg p-3 font-mono text-xs leading-relaxed"
              >
                <code>{block.text}</code>
              </pre>
            )
          default: {
            const _exhaustive: never = block
            return _exhaustive
          }
        }
      })}
    </div>
  )
}

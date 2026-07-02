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
  | { type: 'table'; headers: string[]; rows: string[][] }

function isTableRow(line: string): boolean {
  const trimmed = line.trim()
  if (trimmed.startsWith('|') && trimmed.includes('|', 1)) {
    return true
  }

  const cells = trimmed.split('|').map((cell) => cell.trim())
  return cells.length >= 3 && cells.every((cell) => cell.length > 0)
}

function isTableSeparator(line: string): boolean {
  if (!isTableRow(line)) {
    return false
  }

  return line
    .trim()
    .slice(1)
    .split('|')
    .every((cell) => /^:?-{3,}:?$/.test(cell.trim()))
}

function parseTableRow(line: string): string[] {
  const trimmed = line.trim()
  const inner = trimmed.startsWith('|') ? trimmed.slice(1) : trimmed
  const withoutTrailing = inner.endsWith('|') ? inner.slice(0, -1) : inner
  return withoutTrailing.split('|').map((cell) => cell.trim())
}

function parseMarkdownTable(lines: string[]): { headers: string[]; rows: string[][] } | null {
  if (lines.length < 2) {
    return null
  }

  const headers = parseTableRow(lines[0])
  if (headers.length === 0 || headers.every((cell) => cell.length === 0)) {
    return null
  }

  const bodyStart = isTableSeparator(lines[1]) ? 2 : 1
  const rows = lines.slice(bodyStart).map(parseTableRow).filter((row) => row.some(Boolean))

  if (rows.length === 0 && bodyStart === 1 && !isTableSeparator(lines[1])) {
    return null
  }

  return { headers, rows }
}

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

    if (isTableRow(trimmed)) {
      const tableLines: string[] = []
      while (index < lines.length && isTableRow(lines[index].trim())) {
        tableLines.push(lines[index].trim())
        index += 1
      }

      const table = parseMarkdownTable(tableLines)
      if (table) {
        blocks.push({ type: 'table', headers: table.headers, rows: table.rows })
      } else {
        blocks.push({ type: 'paragraph', text: tableLines.join(' ') })
      }
      continue
    }

    const paragraphLines: string[] = [trimmed]
    index += 1
    while (
      index < lines.length &&
      lines[index].trim().length > 0 &&
      !/^#{1,3}\s+/.test(lines[index].trim()) &&
      !/^[-*]\s+/.test(lines[index].trim()) &&
      !/^\d+\.\s+/.test(lines[index].trim()) &&
      !isTableRow(lines[index].trim())
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
          case 'table':
            return (
              <div key={key} className="overflow-x-auto rounded-lg border border-border/50">
                <table className="w-full min-w-[32rem] border-collapse text-xs">
                  <thead>
                    <tr className="bg-secondary/40 border-border/60 border-b">
                      {block.headers.map((header, headerIndex) => (
                        <th
                          key={`${key}-h-${headerIndex}`}
                          className="text-muted-foreground px-3 py-2 text-left font-medium whitespace-nowrap"
                        >
                          {renderInline(header, `${key}-h-${headerIndex}`)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {block.rows.map((row, rowIndex) => (
                      <tr
                        key={`${key}-r-${rowIndex}`}
                        className="border-border/40 border-b last:border-b-0 even:bg-secondary/15"
                      >
                        {block.headers.map((_, cellIndex) => (
                          <td
                            key={`${key}-r-${rowIndex}-c-${cellIndex}`}
                            className={`px-3 py-2 align-top ${
                              cellIndex === 0
                                ? 'whitespace-normal'
                                : 'whitespace-nowrap tabular-nums'
                            }`}
                          >
                            {renderInline(
                              row[cellIndex] ?? '',
                              `${key}-r-${rowIndex}-c-${cellIndex}`,
                            )}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
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

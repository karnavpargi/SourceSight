/** Normalize SEC filing table text (pipe cells, often collapsed to one line) for markdown rendering. */

function looksLikePipeTable(text: string): boolean {
  const pipeCount = text.match(/\|/g)?.length ?? 0
  return pipeCount >= 3
}

function toGfmTableRow(line: string): string {
  const withoutTrailingPipe = line.replace(/\s*\|\s*$/, '')
  const cells = withoutTrailingPipe.split('|').map((cell) => cell.trim())

  while (cells.length > 0 && cells[cells.length - 1].length === 0) {
    cells.pop()
  }

  if (cells.length < 2) {
    return line
  }

  return `| ${cells.join(' | ')} |`
}

export function normalizePassageMarkdown(text: string): string {
  if (!looksLikePipeTable(text)) {
    return text
  }

  let normalized = text.replace(/\r\n/g, '\n')
  normalized = normalized.replace(/(?<=\d)\s+(?=[A-Za-z(])/g, '\n')

  return normalized
    .split('\n')
    .map((line) => {
      const trimmed = line.trim()
      if (trimmed.length === 0) {
        return ''
      }
      if (!trimmed.includes('|')) {
        return trimmed
      }
      return toGfmTableRow(trimmed)
    })
    .join('\n')
}

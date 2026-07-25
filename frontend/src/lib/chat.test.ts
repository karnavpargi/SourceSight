/// <reference types="node" />
import assert from 'node:assert/strict'
import test from 'node:test'

import {
  formatUsageFooter,
  messageUsage,
  type SourceSightUIMessage,
} from './chat.ts'

test('messageUsage returns the final valid usage part', () => {
  const message = {
    id: 'assistant-1',
    role: 'assistant',
    parts: [
      { type: 'text', text: 'Answer' },
      {
        type: 'data-usage',
        data: {
          input_tokens: 9729,
          output_tokens: 2372,
          estimated_cost_usd: 0.008849,
        },
      },
    ],
  } as SourceSightUIMessage

  assert.deepEqual(messageUsage(message), {
    input_tokens: 9729,
    output_tokens: 2372,
    estimated_cost_usd: 0.008849,
  })
})

test('messageUsage ignores missing and malformed usage parts', () => {
  const oldMessage = {
    id: 'assistant-old',
    role: 'assistant',
    parts: [{ type: 'text', text: 'Old answer' }],
  } as SourceSightUIMessage

  assert.equal(messageUsage(oldMessage), undefined)
})

test('formatUsageFooter formats tokens and a small non-zero cost', () => {
  assert.equal(
    formatUsageFooter({
      input_tokens: 9729,
      output_tokens: 2372,
      estimated_cost_usd: 0.008849,
    }),
    '9.7k input · 2.4k output · ~$0.0088',
  )
})

test('formatUsageFooter omits unavailable cost', () => {
  assert.equal(
    formatUsageFooter({
      input_tokens: 9729,
      output_tokens: 2372,
      estimated_cost_usd: null,
    }),
    '9.7k input · 2.4k output',
  )
})

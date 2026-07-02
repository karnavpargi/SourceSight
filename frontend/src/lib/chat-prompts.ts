export interface ExamplePrompt {
  id: string
  label: string
  prompt: string
}

/** Analyst questions aligned with docs/client-brief.md (sample corpus: AAPL, AMZN, GOOGL, MSFT, NVDA 2021–2025). */
export const EXAMPLE_PROMPTS: ExamplePrompt[] = [
  {
    id: 'apple-revenue-mix',
    label: 'Apple revenue mix shift (2021–2025)',
    prompt:
      "Across Apple's 2021–2025 10-Ks, how did the revenue mix between iPhone, Services, Mac, iPad, and Wearables change, and which category appears to have contributed most to any mix shift?",
  },
  {
    id: 'amazon-aws-segments',
    label: 'Amazon AWS vs segment profitability',
    prompt:
      'For Amazon, compare AWS operating income and margin against North America and International from 2021–2025. In which years did AWS appear to fund losses or weaker profitability elsewhere?',
  },
  {
    id: 'nvidia-data-center',
    label: 'NVIDIA Data Center demand & supply',
    prompt:
      'How did NVIDIA describe demand drivers, customer concentration, and supply constraints for its Data Center business from fiscal 2021 through fiscal 2025?',
  },
  {
    id: 'microsoft-azure-ai',
    label: 'Microsoft Azure & AI infrastructure language',
    prompt:
      "Across Microsoft's 2021–2025 filings, what changed in the way the company describes Azure, AI infrastructure, and cloud capacity constraints?",
  },
  {
    id: 'alphabet-segments',
    label: 'Alphabet revenue segment trends',
    prompt:
      'For Alphabet, how did Google Search, YouTube ads, Google Network, subscriptions/platforms/devices, and Google Cloud revenue trends differ across the available 10-Ks?',
  },
  {
    id: 'risk-factor-changes',
    label: 'AI/cloud/regulatory risk-factor changes',
    prompt:
      'Which of the five companies added, removed, or materially changed risk-factor language related to AI, cloud infrastructure, export controls, supply chain concentration, or regulation between 2021 and 2025?',
  },
  {
    id: 'supplier-concentration',
    label: 'Apple & NVIDIA supplier concentration',
    prompt:
      'For Apple and NVIDIA, what do the filings say about supplier concentration or dependence on third-party manufacturing, and did the wording become more or less urgent over time?',
  },
  {
    id: 'capex-ai-investment',
    label: 'CapEx & AI/cloud investment comparison',
    prompt:
      'Compare capital expenditures and purchase commitments for Microsoft, Alphabet, Amazon, and NVIDIA. What do the filings imply about the scale and timing of AI/cloud infrastructure investment?',
  },
  {
    id: 'geographic-exposure',
    label: 'Geographic revenue exposures',
    prompt:
      'For each company, summarize the most important geographic revenue exposures disclosed in the latest 10-K, then identify any year-over-year changes that could matter to an analyst.',
  },
  {
    id: 'genai-margin-evidence',
    label: 'Gen AI margin claims — evidence vs limits',
    prompt:
      'If an analyst asks whether the filings prove that generative AI improved margins for any of these companies, what evidence exists in the corpus, and where should the bot refuse to infer beyond the filings?',
  },
]

export function filterExamplePrompts(query: string, limit = 8): ExamplePrompt[] {
  const normalized = query.trim().toLowerCase()

  if (normalized.length === 0) {
    return EXAMPLE_PROMPTS.slice(0, limit)
  }

  return EXAMPLE_PROMPTS.filter(
    (item) =>
      item.label.toLowerCase().includes(normalized) ||
      item.prompt.toLowerCase().includes(normalized),
  ).slice(0, limit)
}

const PROMPT_KEY_PREFIX = 'pending-prompt:'

export function stashPendingPrompt(threadId: string, prompt: string): void {
  sessionStorage.setItem(`${PROMPT_KEY_PREFIX}${threadId}`, prompt)
}

export function consumePendingPrompt(threadId: string): string | null {
  const key = `${PROMPT_KEY_PREFIX}${threadId}`
  const prompt = sessionStorage.getItem(key)
  if (prompt !== null) {
    sessionStorage.removeItem(key)
  }
  return prompt
}

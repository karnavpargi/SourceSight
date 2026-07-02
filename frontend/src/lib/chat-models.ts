export type ChatProviderId = 'local' | 'google' | 'opencode'

export interface ChatModelOption {
  id: string
  label: string
}

export interface ChatProviderCatalog {
  id: ChatProviderId
  label: string
  default_model: string
  models: ChatModelOption[]
}

export interface ChatProvidersResponse {
  default_provider: ChatProviderId
  default_model: string
  providers: ChatProviderCatalog[]
}

export interface ChatModelSelection {
  provider: ChatProviderId
  model: string
}

export function initialSelectionFromCatalog(
  catalog: ChatProvidersResponse,
): ChatModelSelection {
  return {
    provider: catalog.default_provider,
    model: catalog.default_model,
  }
}

export function selectionForProvider(
  catalog: ChatProvidersResponse,
  provider: ChatProviderId,
): ChatModelSelection {
  const providerCatalog = catalog.providers.find((entry) => entry.id === provider)
  if (!providerCatalog || providerCatalog.models.length === 0) {
    return initialSelectionFromCatalog(catalog)
  }

  return {
    provider,
    model: providerCatalog.default_model,
  }
}

export function modelsForProvider(
  catalog: ChatProvidersResponse,
  provider: ChatProviderId,
): ChatModelOption[] {
  return catalog.providers.find((entry) => entry.id === provider)?.models ?? []
}

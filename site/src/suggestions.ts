import { useEffect, useState } from 'react'

const ENDPOINT = `${import.meta.env.BASE_URL}api/requests`

/** One submission from the sidebar box, as GET /api/requests returns it (newest first). */
export interface Suggestion {
  path: string
  text: string
  submitted_at: string
}

export type SuggestionsState = { kind: 'loading' } | { kind: 'ok'; items: Suggestion[] } | { kind: 'error'; message: string }

async function fetchSuggestions(): Promise<Suggestion[]> {
  const res = await fetch(ENDPOINT)
  if (res.status === 404 || res.status === 501) throw new Error('served statically; start `uv run python -m jev_tracker.server` for the suggestions list')
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return (await res.json()).items
}

/** Fetched once per mount, so opening the tab shows what is on `main` now. */
export function useSuggestions(): SuggestionsState {
  const [state, setState] = useState<SuggestionsState>({ kind: 'loading' })
  useEffect(() => {
    fetchSuggestions()
      .then((items) => setState({ kind: 'ok', items }))
      .catch((err: unknown) => setState({ kind: 'error', message: err instanceof Error ? err.message : String(err) }))
  }, [])
  return state
}

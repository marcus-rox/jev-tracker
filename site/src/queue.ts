import { useEffect, useState } from 'react'
import type { Queue } from './types'

const ENDPOINT = `${import.meta.env.BASE_URL}api/queue`
export const REFRESH_SECONDS = 60

/** `at` is when the queue was fetched; ages are computed against it so rendering stays pure. */
export type QueueState = { kind: 'loading' } | { kind: 'ok'; queue: Queue; at: Date } | { kind: 'error'; message: string }

async function fetchQueue(): Promise<Queue> {
  const res = await fetch(ENDPOINT)
  if (res.status === 404 || res.status === 501) throw new Error('served statically; start `uv run python -m jev_tracker.server` for the live queue')
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export function useQueue(): QueueState {
  const [state, setState] = useState<QueueState>({ kind: 'loading' })
  useEffect(() => {
    const load = () =>
      fetchQueue()
        .then((queue) => setState({ kind: 'ok', queue, at: new Date() }))
        .catch((err: unknown) => setState({ kind: 'error', message: err instanceof Error ? err.message : String(err) }))
    load()
    const timer = setInterval(load, REFRESH_SECONDS * 1000)
    return () => clearInterval(timer)
  }, [])
  return state
}

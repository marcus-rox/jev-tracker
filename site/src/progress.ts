import { useEffect, useState } from 'react'
import type { ProgressFeed, ShardProgress } from './types'

/** The `progress` web endpoint of the deployed Modal app (`jev_tracker.modal_app.PROGRESS_LABEL`). */
export const PROGRESS_URL = 'https://rox-research--jev-tracker-progress.modal.run'
export const PROGRESS_REFRESH_SECONDS = 10

export type ProgressState = { kind: 'loading' } | { kind: 'ok'; feed: ProgressFeed } | { kind: 'error'; message: string }

async function fetchProgress(): Promise<ProgressFeed> {
  const res = await fetch(PROGRESS_URL)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export function useProgress(enabled: boolean): ProgressState {
  const [state, setState] = useState<ProgressState>({ kind: 'loading' })
  useEffect(() => {
    if (!enabled) return
    const load = () =>
      fetchProgress()
        .then((feed) => setState({ kind: 'ok', feed }))
        .catch((err: unknown) => setState({ kind: 'error', message: err instanceof Error ? err.message : String(err) }))
    load()
    const timer = setInterval(load, PROGRESS_REFRESH_SECONDS * 1000)
    return () => clearInterval(timer)
  }, [enabled])
  return state
}

/** One reranker's bar: its shards summed, with the rate and ETA tqdm would print. */
export interface RerankerBar {
  reranker: string
  done: number
  total: number
  shards: number
  shardsDone: number
  elapsedSeconds: number
  ratePerSecond: number | null
  etaSeconds: number | null
}

/** The bars of the newest experiment started from `config`; [] until a shard has reported. */
export function barsFor(feed: ProgressFeed, config: string): RerankerBar[] {
  const mine = feed.shards.filter((s) => s.config === config)
  if (mine.length === 0) return []
  const newest = mine.reduce((a, b) => (a.started_at > b.started_at ? a : b)).experiment
  const byReranker = new Map<string, ShardProgress[]>()
  for (const s of mine) if (s.experiment === newest) byReranker.set(s.reranker, [...(byReranker.get(s.reranker) ?? []), s])
  return [...byReranker.entries()].map(([reranker, shards]) => rerankerBar(reranker, shards, feed.at))
}

function rerankerBar(reranker: string, shards: ShardProgress[], at: number): RerankerBar {
  const done = shards.reduce((n, s) => n + s.done, 0)
  const total = shards.reduce((n, s) => n + s.total, 0)
  const startedAt = Math.min(...shards.map((s) => s.started_at))
  const lastUpdate = Math.max(...shards.map((s) => s.updated_at))
  const finished = done >= total
  const elapsedSeconds = Math.max(0, (finished ? lastUpdate : at) - startedAt)
  const answeredThisCall = shards.reduce((n, s) => n + (s.done - s.resumed), 0)
  const ratePerSecond = elapsedSeconds > 0 && answeredThisCall > 0 ? answeredThisCall / elapsedSeconds : null
  const etaSeconds = ratePerSecond === null ? null : (total - done) / ratePerSecond
  return { reranker, done, total, shards: shards.length, shardsDone: shards.filter((s) => s.done >= s.total).length, elapsedSeconds, ratePerSecond, etaSeconds }
}

/** tqdm's clock: mm:ss, or h:mm:ss past an hour. */
export function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const ss = s % 60
  const mmss = `${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}`
  return h > 0 ? `${h}:${mmss}` : mmss
}

export function rateText(ratePerSecond: number | null): string {
  if (ratePerSecond === null) return '— req/s'
  if (ratePerSecond >= 1) return `${ratePerSecond.toFixed(1)} req/s`
  return `${(1 / ratePerSecond).toFixed(1)} s/req`
}

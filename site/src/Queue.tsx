import { useState } from 'react'
import { blob, type Decision, type QueueItem } from './types'
import type { Decide, QueueState } from './queue'

const MINUTE_MS = 60_000

function age(fromIso: string, now: Date): string {
  const minutes = Math.max(0, Math.round((now.getTime() - new Date(fromIso).getTime()) / MINUTE_MS))
  if (minutes < 60) return `${minutes} min`
  if (minutes < 1440) return `${Math.round(minutes / 60)} h`
  return `${Math.round(minutes / 1440)} d`
}

function Job({ item, now }: { item: QueueItem; now: Date }) {
  const running = item.status === 'running'
  return (
    <div className={`job ${item.status}`}>
      <div className="m"><span className="dot" />{item.url ? <a href={item.url} target="_blank" rel="noreferrer">{item.label}</a> : item.label}</div>
      <div className="meta">
        {running && item.started_at ? `running · started ${age(item.started_at, now)} ago` : `queued · waiting ${age(item.queued_at, now)}`} · from {item.source}
      </div>
      {running && <div className="bar"><i /></div>}
    </div>
  )
}

/** A model Devin wants approval to run; Approve queues it for the next daily run, Skip drops it. */
function Proposal({ item, now, decide }: { item: QueueItem; now: Date; decide: Decide }) {
  const [pending, setPending] = useState<Decision | null>(null)
  const [error, setError] = useState<string | null>(null)
  const act = (decision: Decision) => {
    setPending(decision)
    setError(null)
    decide(item.config, decision).catch((err: unknown) => {
      setError(err instanceof Error ? err.message : String(err))
      setPending(null)
    })
  }
  return (
    <div className="job proposed">
      <div className="m">{item.url ? <a href={item.url} target="_blank" rel="noreferrer">{item.label}</a> : item.label}</div>
      <div className="meta">proposed {age(item.queued_at, now)} ago · from {item.source}</div>
      {item.note && <div className="note">{item.note}</div>}
      <div className="actions">
        <button type="button" className="approve" disabled={pending !== null} onClick={() => act('approve')}>{pending === 'approve' ? 'Queueing…' : 'Approve — run it'}</button>
        <button type="button" className="reject" disabled={pending !== null} onClick={() => act('reject')}>{pending === 'reject' ? 'Removing…' : 'Skip'}</button>
      </div>
      {error && <div className="meta bad">{error}</div>}
    </div>
  )
}

export function QueueBody({ state, decide }: { state: QueueState; decide: Decide }) {
  if (state.kind === 'loading') return <p className="hint">Loading…</p>
  if (state.kind === 'error') return <p className="hint">Queue unavailable: {state.message}.</p>
  const proposed = state.queue.items.filter((i) => i.status === 'proposed')
  const active = state.queue.items.filter((i) => i.status !== 'proposed')
  if (state.queue.items.length === 0) return <p className="hint">Queue is empty — every runnable model has been evaluated; results are in the table below.</p>
  return (
    <>
      {active.length > 0 && <div className="status">{active.map((item) => <Job key={item.config} item={item} now={state.at} />)}</div>}
      {proposed.length > 0 && (
        <div className="proposals">
          <h3>Awaiting your approval ({proposed.length})</h3>
          <p className="hint">Devin found these but will not run them until you approve. Approve adds the model to the queue for the next daily run (which builds whatever the note says is missing); Skip removes it. Each click is a commit to data/queue.json on main.</p>
          <div className="status">{proposed.map((item) => <Proposal key={item.config} item={item} now={state.at} decide={decide} />)}</div>
        </div>
      )}
    </>
  )
}

export function QueueSub({ state }: { state: QueueState }) {
  const items = state.kind === 'ok' ? state.queue.items : []
  const running = items.filter((i) => i.status === 'running').length
  const proposed = items.filter((i) => i.status === 'proposed').length
  return (
    <>
      {state.kind === 'ok' && <span className="sub">{running} running · {items.length - running - proposed} waiting · {proposed} awaiting approval</span>}
      <span className="sub">{state.kind === 'ok' && '· '}source: <a href={blob('data/queue.json')} target="_blank" rel="noreferrer">data/queue.json</a> · several can run at once · a finished model leaves the queue and appears in the Model list</span>
    </>
  )
}

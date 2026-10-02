import { useState } from 'react'
import { blob, type Decision, type QueueItem, type QueueStatus } from './types'
import type { Decide, QueueState } from './queue'

const MINUTE_MS = 60_000

/** The queue read left to right: a model enters as a proposal, is queued once approved, runs, then leaves for the Model list. */
const STAGES: { status: QueueStatus; title: string; empty: string }[] = [
  { status: 'proposed', title: 'Awaiting your approval', empty: 'nothing to approve' },
  { status: 'queued', title: 'Queued', empty: 'nothing queued' },
  { status: 'running', title: 'Running', empty: 'nothing running' },
]

function age(fromIso: string, now: Date): string {
  const minutes = Math.max(0, Math.round((now.getTime() - new Date(fromIso).getTime()) / MINUTE_MS))
  if (minutes < 60) return `${minutes} min`
  if (minutes < 1440) return `${Math.round(minutes / 60)} h`
  return `${Math.round(minutes / 1440)} d`
}

function since(item: QueueItem, now: Date): string {
  if (item.status === 'running') return item.started_at ? `started ${age(item.started_at, now)} ago` : 'starting'
  return `${item.status === 'proposed' ? 'proposed' : 'queued'} ${age(item.queued_at, now)} ago`
}

/** Approve queues the model for the next daily run (which builds whatever the note says is missing); Skip drops it. */
function Actions({ item, decide }: { item: QueueItem; decide: Decide }) {
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
    <>
      <div className="actions">
        <button type="button" className="approve" disabled={pending !== null} onClick={() => act('approve')}>{pending === 'approve' ? 'Queueing…' : 'Approve'}</button>
        <button type="button" className="reject" disabled={pending !== null} onClick={() => act('reject')}>{pending === 'reject' ? 'Removing…' : 'Skip'}</button>
      </div>
      {error && <div className="meta bad">{error}</div>}
    </>
  )
}

function Row({ item, now, decide }: { item: QueueItem; now: Date; decide: Decide }) {
  return (
    <li className={`row ${item.status}`}>
      <div className="head">
        <span className="dot" />
        <span className="m">{item.url ? <a href={item.url} target="_blank" rel="noreferrer">{item.label}</a> : item.label}</span>
        <span className="meta">{since(item, now)}</span>
      </div>
      <div className="meta">from {item.source}{item.note && <> · <span className="note" title={item.note}>{item.note}</span></>}</div>
      {item.status === 'running' && <div className="bar"><i /></div>}
      {item.status === 'proposed' && <Actions item={item} decide={decide} />}
    </li>
  )
}

export function QueueBody({ state, decide }: { state: QueueState; decide: Decide }) {
  if (state.kind === 'loading') return <p className="hint">Loading…</p>
  if (state.kind === 'error') return <p className="hint">Queue unavailable: {state.message}.</p>
  if (state.queue.items.length === 0) return <p className="hint">Queue is empty — every runnable model has been evaluated; results are in the table below.</p>
  const byStage = STAGES.map((stage) => state.queue.items.filter((item) => item.status === stage.status))
  // A stage with more in it gets more width, capped so an empty stage still reads as a stage.
  const columns = byStage.map((items) => `minmax(0, ${1 + Math.min(items.length, 3)}fr)`).join(' ')
  return (
    <div className="pipeline" style={{ gridTemplateColumns: columns }}>
      {STAGES.map((stage, i) => {
        const items = byStage[i]
        return (
          <div key={stage.status} className="stagewrap">
            {i > 0 && <div className="arrow" aria-hidden="true">→</div>}
            <section className={`stage ${stage.status}`}>
              <h3>{stage.title} <span className="count">{items.length}</span></h3>
              {items.length === 0 ? <p className="hint">{stage.empty}</p> : (
                <ul>{items.map((item) => <Row key={item.config} item={item} now={state.at} decide={decide} />)}</ul>
              )}
            </section>
          </div>
        )
      })}
    </div>
  )
}

export function QueueSub({ state }: { state: QueueState }) {
  const items = state.kind === 'ok' ? state.queue.items : []
  const count = (status: QueueStatus) => items.filter((i) => i.status === status).length
  return (
    <>
      {state.kind === 'ok' && <span className="sub">{count('proposed')} awaiting approval · {count('queued')} queued · {count('running')} running</span>}
      <span className="sub">{state.kind === 'ok' && '· '}source: <a href={blob('data/queue.json')} target="_blank" rel="noreferrer">data/queue.json</a> · Approve / Skip commit to it on main · a finished model leaves the queue and appears in the Model list below</span>
    </>
  )
}

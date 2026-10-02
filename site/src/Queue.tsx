import { blob, type QueueItem } from './types'
import type { QueueState } from './queue'

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

export function QueueBody({ state }: { state: QueueState }) {
  if (state.kind === 'loading') return <p className="hint">Loading…</p>
  if (state.kind === 'error') return <p className="hint">Queue unavailable: {state.message}.</p>
  if (state.queue.items.length === 0) return <p className="hint">Queue is empty — every runnable model has been evaluated; results are in the table below.</p>
  return <div className="status">{state.queue.items.map((item) => <Job key={item.config} item={item} now={state.at} />)}</div>
}

export function QueueSub({ state }: { state: QueueState }) {
  const items = state.kind === 'ok' ? state.queue.items : []
  const running = items.filter((i) => i.status === 'running').length
  return (
    <>
      {state.kind === 'ok' && <span className="sub">{running} running · {items.length - running} waiting</span>}
      <span className="sub">{state.kind === 'ok' && '· '}source: <a href={blob('data/queue.json')} target="_blank" rel="noreferrer">data/queue.json</a> · several can run at once · a finished model leaves the queue and appears in the Model list</span>
    </>
  )
}

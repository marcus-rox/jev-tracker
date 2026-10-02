import { useLayoutEffect, useRef, useState } from 'react'
import { blob, type Evidence, type QueueItem, type QueueStatus } from './types'
import { SKIP_PHRASE, type Decide, type QueueState } from './queue'
import { nextRun, untilText, whenText } from './schedule'
import { useProgress, type ProgressState } from './progress'
import { RunProgress } from './Progress'

const MINUTE_MS = 60_000

/** The queue read left to right: a model enters as a proposal, is queued once approved, runs, then leaves for the Model list. */
const STAGES: { status: QueueStatus; title: string; empty: string }[] = [
  { status: 'proposed', title: 'Awaiting your approval', empty: 'nothing to approve' },
  { status: 'queued', title: 'Queued', empty: 'nothing queued' },
  { status: 'running', title: 'Running', empty: 'nothing running' },
  { status: 'failed', title: 'Failed', empty: 'nothing failed' },
]

function age(fromIso: string, now: Date): string {
  const minutes = Math.max(0, Math.round((now.getTime() - new Date(fromIso).getTime()) / MINUTE_MS))
  if (minutes < 60) return `${minutes} min`
  if (minutes < 1440) return `${Math.round(minutes / 60)} h`
  return `${Math.round(minutes / 1440)} d`
}

function since(item: QueueItem, now: Date): string {
  if (item.status === 'running') return item.started_at ? `started ${age(item.started_at, now)} ago` : 'starting'
  if (item.status === 'failed') return item.finished_at ? `failed ${age(item.finished_at, now)} ago` : 'failed'
  return `${item.status === 'proposed' ? 'proposed' : 'queued'} ${age(item.queued_at, now)} ago`
}

/** Approve queues the model for the next hourly run (which builds whatever the note says is missing); Skip drops it. */
/** Skip is guarded: the phrase typed exactly plus the server's password, then one Skip click. */
function SkipConfirm({ item, decide, onCancel }: { item: QueueItem; decide: Decide; onCancel: () => void }) {
  const [phrase, setPhrase] = useState('')
  const [password, setPassword] = useState('')
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const ready = phrase === SKIP_PHRASE && password.length > 0 && !pending
  const act = () => {
    setPending(true)
    setError(null)
    decide(item.config, 'reject', { phrase, password }).catch((err: unknown) => {
      setError(err instanceof Error ? err.message : String(err))
      setPending(false)
    })
  }
  return (
    <form className="confirm" onSubmit={(e) => { e.preventDefault(); if (ready) act() }}>
      <label>Type <code>{SKIP_PHRASE}</code><input value={phrase} onChange={(e) => setPhrase(e.target.value)} placeholder={SKIP_PHRASE} autoFocus autoComplete="off" spellCheck={false} /></label>
      <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" /></label>
      <div className="actions">
        <button type="submit" className="reject" disabled={!ready}>{pending ? 'Removing…' : 'Skip'}</button>
        <button type="button" disabled={pending} onClick={onCancel}>Cancel</button>
      </div>
      {error && <div className="meta bad">{error}</div>}
    </form>
  )
}

function Actions({ item, decide }: { item: QueueItem; decide: Decide }) {
  const [pending, setPending] = useState(false)
  const [confirming, setConfirming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const approve = () => {
    setPending(true)
    setError(null)
    decide(item.config, 'approve').catch((err: unknown) => {
      setError(err instanceof Error ? err.message : String(err))
      setPending(false)
    })
  }
  if (confirming) return <SkipConfirm item={item} decide={decide} onCancel={() => setConfirming(false)} />
  return (
    <>
      <div className="actions">
        <button type="button" className="approve" disabled={pending} onClick={approve}>{pending ? 'Queueing…' : 'Approve'}</button>
        <button type="button" className="reject" disabled={pending} onClick={() => setConfirming(true)}>Skip</button>
      </div>
      {error && <div className="meta bad">{error}</div>}
    </>
  )
}

/** The Modal-side proof a failed run leaves behind: which calls finished, how many cases each reranker answered. */
function EvidenceList({ evidence }: { evidence: Evidence }) {
  const calls = Object.values(evidence.calls)
  const finished = calls.filter((s) => s === 'finished').length
  return (
    <ul className="evidence">
      <li><b>Experiment</b><a href={blob(`data/experiments/${evidence.experiment}`)} target="_blank" rel="noreferrer">{evidence.experiment}</a></li>
      <li><b>Modal calls</b>{calls.length ? `${finished} of ${calls.length} finished` : 'none recorded'}</li>
      {evidence.rerankers.map((r) => <li key={r.name}><b>{r.name}</b>{r.cases} cases · {r.requests} requests</li>)}
      <li><b>Kept-mass table</b>{evidence.kept_mass ? 'written' : 'missing'}</li>
    </ul>
  )
}

/** Skip drops a failed item once its evidence has been read; same guard as above. */
function Dismiss({ item, decide }: { item: QueueItem; decide: Decide }) {
  const [confirming, setConfirming] = useState(false)
  if (confirming) return <SkipConfirm item={item} decide={decide} onCancel={() => setConfirming(false)} />
  return <div className="actions"><button type="button" className="reject" onClick={() => setConfirming(true)}>Skip</button></div>
}

/** The note is clamped to two lines; the toggle appears only when the clamp actually hides text. */
function Note({ text }: { text: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [clamped, setClamped] = useState(false)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el || open) return
    const update = () => setClamped(el.scrollHeight > el.clientHeight)
    update()
    const observer = new ResizeObserver(update)
    observer.observe(el)
    return () => observer.disconnect()
  }, [text, open])
  return (
    <>
      <div ref={ref} className={`note${open ? ' open' : ''}`}>{text}</div>
      {(clamped || open) && <button type="button" className="more note-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>{open ? 'collapse' : 'expand'}</button>}
    </>
  )
}

const SOURCE_COLOR: Record<string, string> = {
  huggingface: '#f0a94a', github: '#9b6fd0', arxiv: '#c0d86a', web: '#6fcfe8', twitter: '#6fa8ea', hackernews: '#e8864a', slack: '#5ec9a6', submitted: '#b05fb8', manual: '#b05fb8',
}

function Card({ item, now, decide, progress }: { item: QueueItem; now: Date; decide: Decide; progress: ProgressState }) {
  return (
    <li className={`card ${item.status}`}>
      <div className="body">
        <div className="head"><span className="dot" /><span className="m">{item.url ? <a href={item.url} target="_blank" rel="noreferrer">{item.label}</a> : item.label}</span></div>
        <div className="meta">{since(item, now)}</div>
        {item.note && <Note text={item.note} />}
        {item.status === 'running' && <RunProgress config={item.config} state={progress} />}
        {item.status === 'proposed' && <Actions item={item} decide={decide} />}
        {item.status === 'failed' && item.evidence && <EvidenceList evidence={item.evidence} />}
        {item.status === 'failed' && <Dismiss item={item} decide={decide} />}
      </div>
      <div className="foot">
        <span><b>Source</b>{item.source}</span>
        <span className="r"><b>Found</b>{age(item.queued_at, now)} ago</span>
      </div>
      <div className="strip" style={{ background: SOURCE_COLOR[item.source] ?? 'var(--muted)' }}>{item.config.replace(/^configs\//, '').replace(/\.yaml$/, '')}</div>
    </li>
  )
}

/** Queued models run in the next automation run; it fires every hour, so this is when the lane drains. */
function NextRun({ now }: { now: Date }) {
  const at = nextRun(now)
  return (
    <div className="next" title={at.toISOString()}>
      <span className="clock" />next run in <b>{untilText(now, at)}</b><span className="when">{whenText(at)} · hourly</span>
    </div>
  )
}

export function QueueBody({ state, decide }: { state: QueueState; decide: Decide }) {
  const progress = useProgress(state.kind === 'ok' && state.queue.items.some((item) => item.status === 'running'))
  if (state.kind === 'loading') return <p className="hint">Loading…</p>
  if (state.kind === 'error') return <p className="hint">Queue unavailable: {state.message}.</p>
  if (state.queue.items.length === 0) return <p className="hint">Queue is empty — every runnable model has been evaluated; results are in the table below.</p>
  return (
    <div className="board">
      {STAGES.map((stage) => {
        const items = state.queue.items.filter((item) => item.status === stage.status)
        return (
          <section key={stage.status} className={`lane ${stage.status}${items.length ? ' busy' : ''}`}>
            <h3><span className="ring" />{stage.title}<span className="count">{items.length}</span></h3>
            {stage.status === 'queued' && <NextRun now={state.at} />}
            <ul>
              {items.length === 0 ? <li className="hint">{stage.empty}</li> : items.map((item) => <Card key={item.config} item={item} now={state.at} decide={decide} progress={progress} />)}
            </ul>
          </section>
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
      {state.kind === 'ok' && <span className="sub">{count('proposed')} awaiting approval · {count('queued')} queued · {count('running')} running · {count('failed')} failed</span>}
      <span className="sub">{state.kind === 'ok' && '· '}source: <a href={blob('data/queue.json')} target="_blank" rel="noreferrer">data/queue.json</a> · Approve / Skip commit to it on main · a run leaves the queue only once Modal is verified (every call finished, every case answered) and then appears in the Model list below</span>
    </>
  )
}

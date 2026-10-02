import { useMemo, useState } from 'react'
import { blob } from './types'
import type { Suggestion, SuggestionsState } from './suggestions'

const PREVIEW_WORDS = 5
const PREVIEW_CHARS = 60
const URL_RE = /^https?:\/\//

type Dir = 1 | -1

const words = (text: string) => text.split(/\s+/).filter(Boolean)

function preview(text: string): { short: string; cut: boolean } {
  const ws = words(text)
  const head = ws.slice(0, PREVIEW_WORDS).join(' ')
  const short = head.length > PREVIEW_CHARS ? head.slice(0, PREVIEW_CHARS) : head
  return { short, cut: ws.length > PREVIEW_WORDS || short.length < head.length }
}

const when = (iso: string) =>
  new Date(iso).toLocaleString('en-US', { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', timeZoneName: 'short' })

function Linked({ text }: { text: string }) {
  return <>{words(text).map((w, i) => <span key={i}>{i > 0 && ' '}{URL_RE.test(w) ? <a href={w} target="_blank" rel="noreferrer">{w}</a> : w}</span>)}</>
}

function SuggestionRow({ s, open, toggle }: { s: Suggestion; open: boolean; toggle: () => void }) {
  const { short, cut } = preview(s.text)
  return (
    <tr className={open ? 'open' : ''}>
      <td className="when" title={s.submitted_at}>{when(s.submitted_at)}</td>
      <td className="text">
        {open ? <Linked text={s.text} /> : <>{short}{cut && '…'}</>}
        {cut && <button className="more" onClick={toggle} aria-expanded={open}>{open ? 'collapse' : 'expand'}</button>}
      </td>
      <td className="file"><a href={blob(s.path)} target="_blank" rel="noreferrer" title={s.path}>json</a></td>
    </tr>
  )
}

export function SuggestionsSub({ state }: { state: SuggestionsState }) {
  if (state.kind === 'ok') return <>{state.items.length} submitted from the sidebar · <a href={blob('requests')} target="_blank" rel="noreferrer">requests/</a></>
  return <>submitted from the sidebar</>
}

export function SuggestionsBody({ state }: { state: SuggestionsState }) {
  const [dir, setDir] = useState<Dir>(-1)
  const [open, setOpen] = useState(new Set<string>())
  const items = useMemo(() => (state.kind === 'ok' ? state.items : []), [state])
  const sorted = useMemo(() => [...items].sort((a, b) => a.submitted_at.localeCompare(b.submitted_at) * dir), [items, dir])
  if (state.kind === 'loading') return <p className="hint">Loading suggestions…</p>
  if (state.kind === 'error') return <p className="hint err">{state.message}</p>
  if (items.length === 0) return <p className="hint">No suggestions yet. Use the box in the sidebar to add one.</p>
  const toggle = (path: string) => {
    const next = new Set(open)
    if (!next.delete(path)) next.add(path)
    setOpen(next)
  }
  const allOpen = open.size === items.length
  return (
    <>
      <p className="hint">
        Click Submitted to reverse the order. Click expand for the full text.{' '}
        <a href="#" onClick={(e) => { e.preventDefault(); setOpen(allOpen ? new Set() : new Set(items.map((s) => s.path))) }}>{allOpen ? 'Collapse all' : 'Expand all'}</a>
      </p>
      <div className="tablewrap">
        <table className="sugg">
          <thead>
            <tr>
              <th className="when" onClick={() => setDir(dir === -1 ? 1 : -1)} aria-sort={dir === -1 ? 'descending' : 'ascending'}>Submitted<span className="dir">{dir === -1 ? '▼' : '▲'}</span></th>
              <th className="text">Suggestion</th>
              <th className="file">File</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((s) => <SuggestionRow key={s.path} s={s} open={open.has(s.path)} toggle={() => toggle(s.path)} />)}
          </tbody>
        </table>
      </div>
    </>
  )
}

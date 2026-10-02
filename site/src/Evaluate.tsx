import { useState } from 'react'
import { blob } from './types'

type State = { kind: 'idle' } | { kind: 'busy' } | { kind: 'done'; html_url: string } | { kind: 'error'; message: string }

const ENDPOINT = `${import.meta.env.BASE_URL}api/requests`

async function submit(text: string): Promise<string> {
  const res = await fetch(ENDPOINT, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }) })
  if (res.status === 501 || res.status === 405) throw new Error('This page is served statically; start it with `uv run python -m jev_tracker.server` to accept submissions.')
  const body = await res.json().catch(() => ({ error: `${res.status} ${res.statusText}` }))
  if (!res.ok) throw new Error(body.error ?? `${res.status} ${res.statusText}`)
  return body.html_url
}

export default function Evaluate() {
  const [text, setText] = useState('')
  const [state, setState] = useState<State>({ kind: 'idle' })
  const go = async (e: React.FormEvent) => {
    e.preventDefault()
    setState({ kind: 'busy' })
    try {
      setState({ kind: 'done', html_url: await submit(text.trim()) })
      setText('')
    } catch (err) {
      setState({ kind: 'error', message: err instanceof Error ? err.message : String(err) })
    }
  }
  return (
    <form className="eval" onSubmit={go}>
      <p className="hint">
        Paste one link (a Hugging Face model, a GitHub repo, a paper, an API page) or describe the model. It is saved under{' '}
        <a href={blob('requests')} target="_blank" rel="noreferrer">requests/</a>; the next hourly run picks it up and Devin works out how to evaluate it.
      </p>
      <input type="text" value={text} onChange={(e) => setText(e.target.value)} placeholder="text here:" />
      <button type="submit" disabled={state.kind === 'busy' || text.trim() === ''}>{state.kind === 'busy' ? 'Submitting…' : 'Submit'}</button>
      {state.kind === 'done' && <p className="hint">Saved: <a href={state.html_url} target="_blank" rel="noreferrer">{state.html_url}</a></p>}
      {state.kind === 'error' && <p className="hint err">{state.message}</p>}
    </form>
  )
}

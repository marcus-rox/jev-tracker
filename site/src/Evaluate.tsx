import { useState } from 'react'
import { configYaml, issueUrl, METHODS, type Method, type Source } from './evaluate_config'

export default function Evaluate() {
  const [source, setSource] = useState<Source>('kev')
  const [model, setModel] = useState('')
  const [name, setName] = useState('')
  const [keyEnv, setKeyEnv] = useState('')
  const [methods, setMethods] = useState<Method[]>([...METHODS])
  const slug = name.trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '')
  const ready = slug !== '' && model.trim() !== '' && methods.length > 0
  const yaml = ready ? configYaml(slug, source, model.trim(), methods, keyEnv.trim()) : ''
  return (
    <form className="eval" onSubmit={(e) => e.preventDefault()}>
      <p className="note">Fill this in to get the experiment config; "Open issue" opens a prefilled GitHub issue labelled <code>evaluate</code> that the automation picks up. Nothing is sent from this page.</p>
      <label>source</label>
      <select value={source} onChange={(e) => setSource(e.target.value as Source)}>
        <option value="kev">kev — a Hugging Face Kev checkpoint, served in-process on Modal</option>
        <option value="laya">laya — a Laya checkpoint (ModernBERT encoder)</option>
        <option value="api">api — any hosted endpoint that answers the System One request</option>
      </select>
      <label>{source === 'api' ? 'endpoint URL' : 'Hugging Face id (optionally @revision)'}</label>
      <input type="text" value={model} onChange={(e) => setModel(e.target.value)} placeholder={source === 'api' ? 'https://api.example.com/v1/systemone' : 'org/model-name'} />
      <label>model name (becomes the ranker label)</label>
      <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="kev27b_v3" />
      {source === 'api' && <>
        <label>API key environment variable (read from a Modal Secret; never stored here)</label>
        <input type="text" value={keyEnv} onChange={(e) => setKeyEnv(e.target.value)} placeholder="EXAMPLE_API_KEY" />
      </>}
      <label>methods</label>
      <div className="inline">
        {METHODS.map((m) => (
          <label key={m}><input type="checkbox" checked={methods.includes(m)} onChange={() => setMethods(methods.includes(m) ? methods.filter((x) => x !== m) : [...methods, m])} /> {m}</label>
        ))}
      </div>
      {ready && <>
        <label>configs/{slug}.yaml</label>
        <pre className="yaml">{yaml}</pre>
        <a className="btn" href={issueUrl(slug, yaml)} target="_blank" rel="noreferrer">Open issue on GitHub</a>
      </>}
    </form>
  )
}

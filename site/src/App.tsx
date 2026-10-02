import { useEffect, useMemo, useState } from 'react'
import './App.css'
import Evaluate from './Evaluate'
import { blob, type K, type Row, type SiteData } from './types'

type Tab = 'summary' | 'quality' | 'cost' | 'latency' | 'compare' | 'evaluate'
const TABS: [Tab, string][] = [
  ['summary', 'Summary'],
  ['quality', 'Quality'],
  ['cost', 'Cost'],
  ['latency', 'Latency'],
  ['compare', 'Compare runs'],
  ['evaluate', 'Evaluate a new model'],
]

interface Col {
  head: string
  value: (r: Row) => number | null
  fmt: (v: number) => string
  source: (r: Row) => string | null
}

const f = (digits: number) => (v: number) =>
  v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })

const keptCol = (k: K): Col => ({
  head: k === '50' ? 'kept-mass @50' : `@${k}`,
  value: (r) => r.kept_mass?.[k] ?? null,
  fmt: f(3),
  source: (r) => r.sources.kept_mass,
})
const COST: Col[] = [
  { head: 'warm GPU-s', value: (r) => r.cost?.warm_gpu_s ?? null, fmt: f(0), source: (r) => r.sources.cost },
  { head: 'load s', value: (r) => r.cost?.load_s ?? null, fmt: f(0), source: (r) => r.sources.cost },
  { head: '$ / run', value: (r) => r.cost?.warm_usd ?? null, fmt: f(2), source: (r) => r.sources.cost },
  { head: '$ / run with load', value: (r) => r.cost?.in_function_usd ?? null, fmt: f(2), source: (r) => r.sources.cost },
  { head: '$ / query', value: (r) => r.cost?.usd_per_query ?? null, fmt: f(4), source: (r) => r.sources.cost },
  { head: '$ / 1k queries', value: (r) => r.cost?.usd_per_1k ?? null, fmt: f(1), source: (r) => r.sources.cost },
  { head: 'peak GB', value: (r) => r.cost?.peak_gb ?? null, fmt: f(1), source: (r) => r.sources.cost },
]
const LATENCY: Col[] = [
  { head: 'run (s)', value: (r) => r.latency?.run_s ?? null, fmt: f(0), source: (r) => r.sources.latency },
  { head: 's / query', value: (r) => r.latency?.s_per_query ?? null, fmt: f(2), source: (r) => r.sources.latency },
  { head: 'hours / 1k queries', value: (r) => r.latency?.h_per_1k ?? null, fmt: f(2), source: (r) => r.sources.latency },
]
const MEAN: Col = { head: 'mean @50-200', value: (r) => r.mean_kept_mass, fmt: f(3), source: (r) => r.sources.kept_mass }

const LABELS: [string, (r: Row) => string][] = [
  ['ranker', (r) => r.label],
  ['serving', (r) => r.serving],
  ['buffer', (r) => r.buffer],
  ['GPU', (r) => r.gpu],
  ['queries', (r) => String(r.queries)],
]

const servingKind = (r: Row) => r.serving.split(':')[0]
const uniq = (xs: string[]) => [...new Set(xs)]

function Num({ row, col }: { row: Row; col: Col }) {
  const v = col.value(row)
  if (v === null) return <td className="n dim">{row.blank}</td>
  const src = col.source(row)
  return <td className="n">{src ? <a href={blob(src)} title={src} target="_blank" rel="noreferrer">{col.fmt(v)}</a> : col.fmt(v)}</td>
}

type Sort = { key: string; dir: 1 | -1 } | null

function Table({ rows, cols, sort, setSort }: { rows: Row[]; cols: Col[]; sort: Sort; setSort: (s: Sort) => void }) {
  const sorted = useMemo(() => {
    if (!sort) return rows
    const label = LABELS.find(([h]) => h === sort.key)?.[1]
    const col = cols.find((c) => c.head === sort.key)
    return [...rows].sort((a, b) => {
      if (label) return label(a).localeCompare(label(b)) * sort.dir
      const va = col?.value(a) ?? -Infinity, vb = col?.value(b) ?? -Infinity
      return (va - vb) * sort.dir
    })
  }, [rows, cols, sort])
  const click = (key: string) => setSort(sort?.key === key ? (sort.dir === 1 ? { key, dir: -1 } : null) : { key, dir: 1 })
  const arrow = (key: string) => sort?.key === key ? <span className="dir">{sort.dir === 1 ? '▲' : '▼'}</span> : null
  const group = (r: Row) => `${r.experiment}|${servingKind(r)}|${r.gpu}`
  return (
    <table>
      <thead>
        <tr>
          {LABELS.map(([h]) => <th key={h} onClick={() => click(h)}>{h}{arrow(h)}</th>)}
          {cols.map((c) => <th key={c.head} className="n" onClick={() => click(c.head)}>{c.head}{arrow(c.head)}</th>)}
        </tr>
      </thead>
      <tbody>
        {sorted.map((r, i) => (
          <tr key={`${r.experiment}/${r.reranker}`} className={[r.highlight && 'hl', !sort && i > 0 && group(r) !== group(sorted[i - 1]) && 'sep'].filter(Boolean).join(' ')}>
            {LABELS.map(([h, get], j) => <td key={h}>{j === 0 ? <a href={blob(r.sources.config)} title={r.experiment} target="_blank" rel="noreferrer">{get(r)}</a> : get(r)}</td>)}
            {cols.map((c) => <Num key={c.head} row={r} col={c} />)}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** Each family's 75-query row with the highest mean kept-mass (first wins a tie), as in the report. */
function bestPerFamily(rows: Row[]): Row[] {
  const best = new Map<string, Row>()
  for (const r of rows) {
    if (r.queries !== 75 || r.mean_kept_mass === null) continue
    const b = best.get(r.family)
    if (!b || r.mean_kept_mass > b.mean_kept_mass!) best.set(r.family, r)
  }
  return [...best.values()].sort((a, b) => b.mean_kept_mass! - a.mean_kept_mass!)
}

function Compare({ rows, experiments, ks }: { rows: Row[]; experiments: string[]; ks: K[] }) {
  if (experiments.length < 2) return <p className="note">Tick two or more experiments under "Compare" in the sidebar.</p>
  const labels = uniq(rows.filter((r) => experiments.includes(r.experiment)).map((r) => r.label))
  const cell = (label: string, exp: string) => rows.find((r) => r.label === label && r.experiment === exp)
  const cols: [string, Col][] = [...ks.map((k) => [`@${k}`, keptCol(k)] as [string, Col]), ['$ / 1k', COST[5]], ['s / query', LATENCY[1]]]
  return (
    <table className="cmp">
      <thead>
        <tr><th rowSpan={2}>ranker</th>{experiments.map((e) => <th key={e} className="exp" colSpan={cols.length}>{e}</th>)}</tr>
        <tr>{experiments.flatMap((e) => cols.map(([h]) => <th key={e + h} className="n">{h}</th>))}</tr>
      </thead>
      <tbody>
        {labels.map((label) => (
          <tr key={label}>
            <td>{label}</td>
            {experiments.flatMap((e) => {
              const r = cell(label, e)
              return cols.map(([h, c]) => r ? <Num key={e + h} row={r} col={c} /> : <td key={e + h} className="n dim">·</td>)
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Checks({ title, options, on, toggle, mono }: { title: string; options: string[]; on: Set<string>; toggle: (v: string) => void; mono?: boolean }) {
  return (
    <>
      <h3>{title}</h3>
      {options.map((o) => (
        <label key={o} className={mono ? 'exp' : ''} title={o}>
          <input type="checkbox" checked={on.has(o)} onChange={() => toggle(o)} />{o}
        </label>
      ))}
    </>
  )
}

const toggled = (s: Set<string>, v: string) => {
  const n = new Set(s)
  if (!n.delete(v)) n.add(v)
  return n
}

export default function App() {
  const [data, setData] = useState<SiteData | null>(null)
  const [tab, setTab] = useState<Tab>('summary')
  const [families, setFamilies] = useState(new Set<string>())
  const [servings, setServings] = useState(new Set<string>())
  const [gpus, setGpus] = useState(new Set<string>())
  const [exps, setExps] = useState(new Set<string>())
  const [only75, setOnly75] = useState(true)
  const [ks, setKs] = useState(new Set<string>(['50', '100', '150', '200']))
  const [compare, setCompare] = useState(new Set<string>())
  const [sort, setSort] = useState<Sort>(null)

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/rows.json`).then((r) => r.json()).then(setData)
  }, [])
  if (!data) return <p className="note">Loading data/rows.json…</p>

  const all = data.rows
  const pass = (r: Row) =>
    (families.size === 0 || families.has(r.family)) &&
    (servings.size === 0 || servings.has(servingKind(r))) &&
    (gpus.size === 0 || gpus.has(r.gpu)) &&
    (exps.size === 0 || exps.has(r.experiment)) &&
    (!only75 || r.queries === 75)
  const rows = all.filter(pass)
  const shownKs = data.ks.filter((k) => ks.has(k))
  const kept = shownKs.map(keptCol)
  const forTable = (t: string) => rows.filter((r) => r.tables.includes(t))
  const experiments = Object.keys(data.experiments).sort()
  const reset = () => { setFamilies(new Set()); setServings(new Set()); setGpus(new Set()); setExps(new Set()); setOnly75(true); setSort(null) }

  return (
    <div className="app">
      <aside className="side">
        <h1>jev-tracker</h1>
        <p className="sub">{all.length} rows · {experiments.length} experiments · 75 frozen queries</p>
        <button onClick={reset}>reset filters</button>
        <h3>k columns</h3>
        {data.ks.map((k) => <label key={k}><input type="checkbox" checked={ks.has(k)} onChange={() => setKs(toggled(ks, k))} />@{k}</label>)}
        <h3>queries</h3>
        <label><input type="checkbox" checked={only75} onChange={() => setOnly75(!only75)} />75-query runs only</label>
        <Checks title="model family" options={uniq(all.map((r) => r.family))} on={families} toggle={(v) => setFamilies(toggled(families, v))} />
        <Checks title="serving" options={uniq(all.map(servingKind))} on={servings} toggle={(v) => setServings(toggled(servings, v))} />
        <Checks title="GPU" options={uniq(all.map((r) => r.gpu))} on={gpus} toggle={(v) => setGpus(toggled(gpus, v))} />
        <Checks title="experiment" options={experiments} on={exps} toggle={(v) => setExps(toggled(exps, v))} mono />
        <Checks title="compare (2+)" options={experiments} on={compare} toggle={(v) => setCompare(toggled(compare, v))} mono />
      </aside>
      <main className="main">
        <nav className="tabs">
          {TABS.map(([t, name]) => <button key={t} className={tab === t ? 'on' : ''} onClick={() => { setTab(t); setSort(null) }}>{name}</button>)}
        </nav>
        <p className="note">Click any column header to sort; click again to reverse, a third time to clear.</p>
        {tab === 'summary' && <>
          <p className="note">Each model family's 75-query ranker with the highest mean kept-mass over @50–200, with that run's cost and latency. Click a number for its source JSON.</p>
          <Table rows={bestPerFamily(rows)} cols={[...kept, MEAN, COST[5], LATENCY[1]]} sort={sort} setSort={setSort} />
        </>}
        {tab === 'quality' && <Table rows={forTable('quality')} cols={kept} sort={sort} setSort={setSort} />}
        {tab === 'cost' && <Table rows={forTable('cost')} cols={COST} sort={sort} setSort={setSort} />}
        {tab === 'latency' && <Table rows={forTable('latency')} cols={LATENCY} sort={sort} setSort={setSort} />}
        {tab === 'compare' && <Compare rows={rows} experiments={experiments.filter((e) => compare.has(e))} ks={shownKs} />}
        {tab === 'evaluate' && <Evaluate />}
      </main>
    </div>
  )
}

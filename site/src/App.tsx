import { useEffect, useMemo, useState, type ReactNode } from 'react'
import './App.css'
import { HBars, LineChart } from './Chart'
import Dropdown from './Dropdown'
import Evaluate from './Evaluate'
import { QueueBody, QueueSub } from './Queue'
import { REFRESH_SECONDS, useQueue } from './queue'
import { SuggestionsBody, SuggestionsSub } from './Suggestions'
import { useSuggestions } from './suggestions'
import Table, { type Col, type Sort } from './Table'
import Tldr from './Tldr'
import ThemeSelect from './Theme'
import { REPO, blob, type Card, type K, type Row, type SiteData } from './types'

type Tab = 'summary' | 'quality' | 'cost' | 'latency' | 'evaluate' | 'suggestions'
const TABS: [Tab, string][] = [
  ['summary', 'Summary'],
  ['quality', 'Quality'],
  ['cost', 'Cost'],
  ['latency', 'Latency'],
  ['evaluate', 'Jevs to Process'],
  ['suggestions', 'Suggestions'],
]
const FROZEN_QUERIES = 75
const CHART_SUB = 'Top 10 shown (best run per model, Jev / production / random always included) · full list in the table · click a legend entry to hide it'

const f = (digits: number) => (v: number) => v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })

const RATING = 'A rating is the mean of 3 grader votes, each 0 to 3.'
const keptCol = (k: K): Col => ({
  head: `@${k}`, value: (r) => r.kept_mass?.[k] ?? null, fmt: f(3), source: (r) => r.sources.kept_mass,
  help: `kept-mass@${k}. For each query: add up the ratings of the ${k} children this ranker puts on top, then divide by the best sum any ${k} children could reach (${k} is capped at the query's labeled children). Averaged over the queries; 1.0 means as good as the ideal order. ${RATING}`,
})
const MEAN: Col = {
  head: 'mean', value: (r) => r.mean_kept_mass, fmt: f(3), source: (r) => r.sources.kept_mass,
  help: 'Conglomerate quality score: the average of kept-mass @50, @100, @150 and @200. The best-quality card, the "quality > prod" filter and the quality sort use this number.',
}
const API_ROWS = 'Production and Jev rows use their hosted-API bill for the run.'
const COST: Col[] = [
  { head: 'warm GPU-s', value: (r) => r.cost?.warm_gpu_s ?? null, fmt: f(0), source: (r) => r.sources.cost,
    help: 'GPU-seconds spent answering. For each GPU: time from model loaded to its last answer, at full load (16 requests in flight per GPU by default). Summed over every GPU in the run. Model loading is not included. Blank for hosted-API rows.' },
  { head: 'load s', value: (r) => r.cost?.load_s ?? null, fmt: f(0), source: (r) => r.sources.cost,
    help: 'Start-up seconds: from container start to model loaded and ready, summed over every GPU in the run. Not included in $ / run.' },
  { head: '$ / run', value: (r) => r.cost?.warm_usd ?? null, fmt: f(2), source: (r) => r.sources.cost,
    help: `Cost of one run (every query once): warm GPU-s times Modal's list price per second for that GPU, plus the reserved host RAM per GiB-second. Load time, failed calls, CPU and network are not counted. ${API_ROWS}` },
  { head: '$ / run + load', value: (r) => r.cost?.in_function_usd ?? null, fmt: f(2), source: (r) => r.sources.cost,
    help: `The same prices applied to each GPU's whole function time (start-up, answering and shutdown), summed over every GPU. ${API_ROWS}` },
  { head: '$ / query', value: (r) => r.cost?.usd_per_query ?? null, fmt: f(4), source: (r) => r.sources.cost,
    help: '$ / run divided by the number of queries in the run.' },
  { head: '$ / 1k queries', value: (r) => r.cost?.usd_per_1k ?? null, fmt: f(1), source: (r) => r.sources.cost,
    help: '$ / run times 1000, divided by the number of queries in the run (straight-line scaling). The cheapest card, the "cost < prod" filter and the cost sort use this number.' },
  { head: 'peak GB', value: (r) => r.cost?.peak_gb ?? null, fmt: f(1), source: (r) => r.sources.cost,
    help: 'The most GPU memory allocated at any moment while answering; the largest value over the run\'s GPUs.' },
]
const WALL = 'Per-query wall clock: the time from sending a query\'s first request to receiving its last answer.'
const LATENCY: Col[] = [
  { head: 's / query (mean)', value: (r) => r.latency?.s_per_query ?? null, fmt: f(2), source: (r) => r.sources.latency,
    help: `${WALL} This column is the average over the queries. Fan-out GPU rows run one query at a time, with its requests spread over a pool of GPUs, after 3 untimed warm-up queries. GPU rows without a fan-out run were timed inside the full-load run, with other queries in flight on the same GPU, so they read higher. Production and Jev rows use the same measurement against the hosted API. The fastest card, the "latency < prod" filter and the latency sort use this number.` },
  { head: 'median', value: (r) => r.latency?.p50_s ?? null, fmt: f(2), source: (r) => r.sources.latency,
    help: `${WALL} This column is the median over the queries.` },
  { head: 'p95', value: (r) => r.latency?.p95_s ?? null, fmt: f(2), source: (r) => r.sources.latency,
    help: `${WALL} This column is the 95th percentile over the queries: about 95% of queries finished within this time.` },
]
const COST_PER_1K = COST[5], S_PER_QUERY = LATENCY[0]

/** Quality is the conglomerate kept-mass: the mean over every k (@50 … @200), so no single k picks the winner. */
type Metric = 'quality' | 'cost' | 'latency'
const METRICS: [Metric, string, Col, 1 | -1][] = [
  ['quality', 'quality (mean kept-mass)', MEAN, -1],
  ['cost', 'cost ($ / 1k)', COST_PER_1K, 1],
  ['latency', 'latency (s / query)', S_PER_QUERY, 1],
]
const BOUNDS = METRICS
const PROD_FAMILY = 'production'

/** A zero cost or latency means the run was never timed, not that it was free. */
const measured = (v: number | null) => (v === null || v === 0 ? null : v)
const beatsProd = (r: Row, prod: Row, col: Col, dir: 1 | -1) => {
  const v = measured(col.value(r)), p = measured(col.value(prod))
  return v !== null && p !== null && (v - p) * dir < 0
}
const ranked = (rows: Row[], metric: Metric | null) => {
  const m = METRICS.find(([k]) => k === metric)
  if (!m) return rows
  const [, , col, dir] = m
  const missing = dir === 1 ? Infinity : -Infinity
  return [...rows].sort((a, b) => ((measured(col.value(a)) ?? missing) - (measured(col.value(b)) ?? missing)) * dir)
}

const uniq = (xs: string[]) => [...new Set(xs)]
const toggled = (s: Set<string>, v: string) => {
  const n = new Set(s)
  if (!n.delete(v)) n.add(v)
  return n
}

function Widget({ span, title, sub, children }: { span: 3 | 4 | 6 | 8 | 9 | 12; title: string; sub?: ReactNode; children: ReactNode }) {
  return (
    <section className={`w s${span}`}>
      <h2>{title}{sub && <span className="sub">{sub}</span>}</h2>
      {children}
    </section>
  )
}

/** Card detail is "<who> · Jev <value>"; the label is "<name> (<unit>)". */
function NumWidget({ card, higherIsBetter }: { card: Card; higherIsBetter: boolean }) {
  const [who, jev] = card.detail.split(' · Jev ')
  const v = parseFloat(card.value), j = parseFloat(jev)
  const rel = (v - j) / j
  const good = higherIsBetter ? v >= j : v <= j
  const unit = card.label.match(/\((.*)\)/)?.[1] ?? ''
  return (
    <Widget span={4} title={card.label.split(' (')[0]}>
      <div className="num">
        <div className="v">{card.value}<span className={`delta ${good ? 'good' : 'bad'}`}>{rel >= 0 ? '+' : ''}{(100 * rel).toFixed(0)}% vs Jev</span></div>
        <div className="who">{who}</div>
        <div className="cmp">{unit} · Jev {jev}</div>
      </div>
    </Widget>
  )
}

/** Mounted only while its tab is open, so each visit re-reads requests/ from the server. */
function SuggestionsWidget() {
  const state = useSuggestions()
  return <Widget span={12} title="Suggestions" sub={<SuggestionsSub state={state} />}><SuggestionsBody state={state} /></Widget>
}

export default function App() {
  const [data, setData] = useState<SiteData | null>(null)
  const [tab, setTab] = useState<Tab>('summary')
  const [models, setModels] = useState<Set<string> | null>(null)
  const [gpus, setGpus] = useState<Set<string> | null>(null)
  const [runtimes, setRuntimes] = useState<Set<string> | null>(null)
  const [hidden, setHidden] = useState(new Set<string>())
  const [sort, setSort] = useState<Sort>(null)
  const [bounds, setBounds] = useState(new Set<string>())
  const [metric, setMetric] = useState<Metric | null>(null)
  const [queue, decide] = useQueue()
  const [side, setSide] = useState(() => localStorage.getItem('side') !== 'closed')
  const toggleSide = () => { localStorage.setItem('side', side ? 'closed' : 'open'); setSide(!side) }

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/rows.json`, { cache: 'no-cache' }).then((r) => r.json()).then(setData)
  }, [])
  const all = useMemo(() => (data?.rows ?? []).filter((r) => r.queries === FROZEN_QUERIES), [data])
  const MODELS = useMemo(() => uniq(all.map((r) => r.label)).sort((a, b) => a.localeCompare(b)), [all])
  const GPUS = useMemo(() => uniq(all.map((r) => r.gpu)), [all])
  const RUNTIMES = useMemo(() => uniq(all.map((r) => r.runtime)).sort((a, b) => a.localeCompare(b)), [all])
  if (!data) return <p className="hint" style={{ padding: 20 }}>Loading data/rows.json…</p>

  const modelSet = models ?? new Set(MODELS)
  const gpuSet = gpus ?? new Set(GPUS)
  const runtimeSet = runtimes ?? new Set(RUNTIMES)
  const prod = all.find((r) => r.family === PROD_FAMILY) ?? null
  const active = BOUNDS.filter(([m]) => bounds.has(m))
  const inBounds = (r: Row) => r.family === PROD_FAMILY || prod === null || active.every(([, , col, dir]) => beatsProd(r, prod, col, dir))
  const rows = ranked(all.filter((r) => modelSet.has(r.label) && gpuSet.has(r.gpu) && runtimeSet.has(r.runtime) && inBounds(r)), metric)
  const forTable = (t: string) => rows.filter((r) => r.tables.includes(t))
  const kept = data.ks.map(keptCol)
  const reset = () => { setModels(null); setGpus(null); setRuntimes(null); setHidden(new Set()); setSort(null); setBounds(new Set()); setMetric(null) }
  const sortTable = (s: Sort) => { setSort(s); setMetric(null) }
  const pickMetric = (m: Metric) => { setMetric(metric === m ? null : m); setSort(null) }
  const prodValue = (col: Col) => {
    const p = prod === null ? null : measured(col.value(prod))
    return p === null ? 'production was never timed' : `production: ${col.fmt(p)} ${col.head}`
  }
  const toggleHidden = (family: string) => setHidden(toggled(hidden, family))
  const chart = (height: number) => <LineChart rows={rows} ks={data.ks} height={height} hidden={hidden} onToggle={toggleHidden} />
  const costBars = <HBars rows={rows} value={(r) => r.cost?.usd_per_1k} digits={2} />
  const latencyBars = <HBars rows={rows} value={(r) => r.latency?.s_per_query} digits={2} />

  return (
    <>
      <header className="topbar">
        <span className="brand">jev-tracker</span>
        <span className="crumb">Dashboards › Jev alternatives · {FROZEN_QUERIES} frozen queries</span>
        <div className="right">
          <span>Last updated {data.updated}</span>
          <span className="pill" title={`queue refreshes every ${REFRESH_SECONDS} s`}>↻ {REFRESH_SECONDS} s</span>
          <ThemeSelect />
          <a href={REPO} target="_blank" rel="noreferrer">GitHub</a>
        </div>
      </header>
      <div className={`layout${side ? '' : ' closed'}`}>
      <aside className="side">
        <Widget span={12} title="Suggest a model to scrape"><Evaluate /></Widget>
      </aside>
      <button className="rail" onClick={toggleSide} title={side ? 'Hide the suggestion panel' : 'Show the suggestion panel'} aria-expanded={side}>
        <span className="arrow">{side ? '‹' : '›'}</span>
      </button>
      <div className="content">
      <div className="toolbar">
        <Dropdown name="Model" options={MODELS} selected={modelSet} onChange={setModels} />
        <Dropdown name="GPU" options={GPUS} selected={gpuSet} onChange={setGpus} />
        <Dropdown name="Runtime" options={RUNTIMES} selected={runtimeSet} onChange={setRuntimes} />
        <span className="label">beats prod</span>
        {BOUNDS.map(([m, , col, dir]) => (
          <span key={m} className={`chip${bounds.has(m) ? ' on' : ''}`} title={prodValue(col)} onClick={() => setBounds(toggled(bounds, m))}>{m} {dir === 1 ? '<' : '>'} prod</span>
        ))}
        <span className="label">sort by</span>
        {METRICS.map(([m, name, , dir]) => (
          <span key={m} className={`chip${metric === m ? ' on' : ''}`} title={dir === 1 ? 'lowest first' : 'highest first'} onClick={() => pickMetric(m)}>{name}</span>
        ))}
        <span className="chip reset" onClick={reset}>reset</span>
        <span className="label">{rows.length} of {all.length} runs</span>
      </div>
      <nav className="tabs">
        {TABS.map(([t, name]) => <button key={t} className={tab === t ? 'on' : ''} onClick={() => { setTab(t); setSort(null) }}>{name}</button>)}
      </nav>
      <main className="grid">
        {tab === 'summary' && <>
          {data.cards.map((c, i) => <NumWidget key={c.label} card={c} higherIsBetter={i === 0} />)}
          <Widget span={3} title="TLDR" sub={<>written by the hourly run · <a href={blob('data/tldr.md')} target="_blank" rel="noreferrer">data/tldr.md</a></>}>
            <Tldr text={data.tldr} />
          </Widget>
          <Widget span={9} title="Quality · kept-mass@k" sub={CHART_SUB}>{chart(360)}</Widget>
          <Widget span={6} title="Cost · $ per 1k queries" sub="cheapest run per family">{costBars}</Widget>
          <Widget span={6} title="Latency · seconds per query" sub="fastest run per family">{latencyBars}</Widget>
          <Widget span={12} title="All runs" sub={`${rows.length} rows · ${FROZEN_QUERIES} frozen queries`}>
            <Table rows={rows} cols={[...kept, MEAN, COST_PER_1K, S_PER_QUERY]} sort={sort} setSort={sortTable} />
          </Widget>
        </>}
        {tab === 'quality' && <>
          <Widget span={12} title="Quality · kept-mass@k" sub={CHART_SUB}>{chart(360)}</Widget>
          <Widget span={12} title="Quality · all runs"><Table rows={forTable('quality')} cols={kept} sort={sort} setSort={sortTable} /></Widget>
        </>}
        {tab === 'cost' && <>
          <Widget span={12} title="Cost · $ per 1k queries" sub="cheapest run per family">{costBars}</Widget>
          <Widget span={12} title="Cost · all runs"><Table rows={forTable('cost')} cols={COST} sort={sort} setSort={sortTable} /></Widget>
        </>}
        {tab === 'latency' && <>
          <Widget span={12} title="Latency · seconds per query" sub="fastest run per family">{latencyBars}</Widget>
          <Widget span={12} title="Latency · all runs"><Table rows={forTable('latency')} cols={LATENCY} sort={sort} setSort={sortTable} /></Widget>
        </>}
        {tab === 'suggestions' && <SuggestionsWidget />}
        {tab === 'evaluate' && <Widget span={12} title="Model evaluation sprint" sub={<QueueSub state={queue} />}><QueueBody state={queue} decide={decide} /></Widget>}
      </main>
      </div>
      </div>
      <p className="foot">{data.rows.length} rows · {Object.keys(data.experiments).length} experiments · data regenerated by the hourly run from the committed experiments.</p>
    </>
  )
}

import { useEffect, useMemo, useState, type ReactNode } from 'react'
import './App.css'
import { HBars, LineChart } from './Chart'
import Dropdown from './Dropdown'
import Evaluate from './Evaluate'
import { QueueBody, QueueSub } from './Queue'
import { REFRESH_SECONDS, useQueue } from './queue'
import Table, { type Col, type Sort } from './Table'
import ThemeSelect from './Theme'
import { REPO, blob, type Card, type K, type SiteData } from './types'

type Tab = 'summary' | 'quality' | 'cost' | 'latency' | 'evaluate'
const TABS: [Tab, string][] = [
  ['summary', 'Summary'],
  ['quality', 'Quality'],
  ['cost', 'Cost'],
  ['latency', 'Latency'],
  ['evaluate', 'Evaluate a new model'],
]
const FROZEN_QUERIES = 75
const CHART_SUB = 'Top 10 shown (best run per model, Jev / production / random always included) · full list in the table · click a legend entry to hide it'

const f = (digits: number) => (v: number) => v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })

const keptCol = (k: K): Col => ({ head: `@${k}`, value: (r) => r.kept_mass?.[k] ?? null, fmt: f(3), source: (r) => r.sources.kept_mass })
const MEAN: Col = { head: 'mean', value: (r) => r.mean_kept_mass, fmt: f(3), source: (r) => r.sources.kept_mass }
const COST: Col[] = [
  { head: 'warm GPU-s', value: (r) => r.cost?.warm_gpu_s ?? null, fmt: f(0), source: (r) => r.sources.cost },
  { head: 'load s', value: (r) => r.cost?.load_s ?? null, fmt: f(0), source: (r) => r.sources.cost },
  { head: '$ / run', value: (r) => r.cost?.warm_usd ?? null, fmt: f(2), source: (r) => r.sources.cost },
  { head: '$ / run + load', value: (r) => r.cost?.in_function_usd ?? null, fmt: f(2), source: (r) => r.sources.cost },
  { head: '$ / query', value: (r) => r.cost?.usd_per_query ?? null, fmt: f(4), source: (r) => r.sources.cost },
  { head: '$ / 1k queries', value: (r) => r.cost?.usd_per_1k ?? null, fmt: f(1), source: (r) => r.sources.cost },
  { head: 'peak GB', value: (r) => r.cost?.peak_gb ?? null, fmt: f(1), source: (r) => r.sources.cost },
]
const LATENCY: Col[] = [
  { head: 'run (s)', value: (r) => r.latency?.run_s ?? null, fmt: f(0), source: (r) => r.sources.latency },
  { head: 's / query', value: (r) => r.latency?.s_per_query ?? null, fmt: f(2), source: (r) => r.sources.latency },
  { head: 'hours / 1k', value: (r) => r.latency?.h_per_1k ?? null, fmt: f(2), source: (r) => r.sources.latency },
]
const COST_PER_1K = COST[5], S_PER_QUERY = LATENCY[1]

const uniq = (xs: string[]) => [...new Set(xs)]
const toggled = (s: Set<string>, v: string) => {
  const n = new Set(s)
  if (!n.delete(v)) n.add(v)
  return n
}

function Widget({ span, title, sub, children }: { span: 4 | 6 | 8 | 12; title: string; sub?: ReactNode; children: ReactNode }) {
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

export default function App() {
  const [data, setData] = useState<SiteData | null>(null)
  const [tab, setTab] = useState<Tab>('summary')
  const [families, setFamilies] = useState<Set<string> | null>(null)
  const [gpus, setGpus] = useState<Set<string> | null>(null)
  const [hidden, setHidden] = useState(new Set<string>())
  const [sort, setSort] = useState<Sort>(null)
  const [queue, decide] = useQueue()

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/rows.json`).then((r) => r.json()).then(setData)
  }, [])
  const all = useMemo(() => (data?.rows ?? []).filter((r) => r.queries === FROZEN_QUERIES), [data])
  const FAMS = useMemo(() => uniq(all.map((r) => r.family)), [all])
  const GPUS = useMemo(() => uniq(all.map((r) => r.gpu)), [all])
  if (!data) return <p className="hint" style={{ padding: 20 }}>Loading data/rows.json…</p>

  const fams = families ?? new Set(FAMS)
  const gpuSet = gpus ?? new Set(GPUS)
  const rows = all.filter((r) => fams.has(r.family) && gpuSet.has(r.gpu))
  const forTable = (t: string) => rows.filter((r) => r.tables.includes(t))
  const kept = data.ks.map(keptCol)
  const reset = () => { setFamilies(null); setGpus(null); setHidden(new Set()); setSort(null) }
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
      <div className="toolbar">
        <Dropdown name="Model" options={FAMS} selected={fams} onChange={setFamilies} />
        <Dropdown name="GPU" options={GPUS} selected={gpuSet} onChange={setGpus} />
        <span className="chip reset" onClick={reset}>reset</span>
        <span className="label">{rows.length} of {all.length} runs</span>
      </div>
      <nav className="tabs">
        {TABS.map(([t, name]) => <button key={t} className={tab === t ? 'on' : ''} onClick={() => { setTab(t); setSort(null) }}>{name}</button>)}
      </nav>
      <main className="grid">
        {tab === 'summary' && <>
          {data.cards.map((c, i) => <NumWidget key={c.label} card={c} higherIsBetter={i === 0} />)}
          <Widget span={8} title="Quality · kept-mass@k" sub={CHART_SUB}>{chart(300)}</Widget>
          <Widget span={4} title="TLDR" sub={<>written by the daily run · <a href={blob('data/tldr.md')} target="_blank" rel="noreferrer">data/tldr.md</a></>}>
            <div className="tldr">{data.tldr.split('\n').filter((line) => line.trim() !== '').map((line, i) => <p key={i}>{line}</p>)}</div>
          </Widget>
          <Widget span={6} title="Cost · $ per 1k queries" sub="cheapest run per family">{costBars}</Widget>
          <Widget span={6} title="Latency · seconds per query" sub="fastest run per family">{latencyBars}</Widget>
          <Widget span={12} title="All runs" sub={`${rows.length} rows · ${FROZEN_QUERIES} frozen queries`}>
            <Table rows={rows} cols={[...kept, MEAN, COST_PER_1K, S_PER_QUERY]} sort={sort} setSort={setSort} />
          </Widget>
        </>}
        {tab === 'quality' && <>
          <Widget span={12} title="Quality · kept-mass@k" sub={CHART_SUB}>{chart(360)}</Widget>
          <Widget span={12} title="Quality · all runs"><Table rows={forTable('quality')} cols={kept} sort={sort} setSort={setSort} /></Widget>
        </>}
        {tab === 'cost' && <>
          <Widget span={12} title="Cost · $ per 1k queries" sub="cheapest run per family">{costBars}</Widget>
          <Widget span={12} title="Cost · all runs"><Table rows={forTable('cost')} cols={COST} sort={sort} setSort={setSort} /></Widget>
        </>}
        {tab === 'latency' && <>
          <Widget span={12} title="Latency · seconds per query" sub="fastest run per family">{latencyBars}</Widget>
          <Widget span={12} title="Latency · all runs"><Table rows={forTable('latency')} cols={LATENCY} sort={sort} setSort={setSort} /></Widget>
        </>}
        {tab === 'evaluate' && <>
          <Widget span={12} title="Model evaluation queue" sub={<QueueSub state={queue} />}><QueueBody state={queue} decide={decide} /></Widget>
          <Widget span={4} title="Suggest a model to scrape"><Evaluate /></Widget>
        </>}
      </main>
      <p className="foot">{data.rows.length} rows · {Object.keys(data.experiments).length} experiments · data regenerated by the daily run from the committed experiments.</p>
    </>
  )
}

import type { K, Row } from './types'

const TOP_N = 10
const REFS = new Set(['jev', 'production', 'random'])
// Okabe-Ito colorblind-safe palette; every series also gets its own marker shape and dash so colour is never the only cue.
const PALETTE = ['#0072b2', '#e69f00', '#009e73', '#d55e00', '#56b4e9', '#cc79a7', '#b8a400']
const MARKERS = ['circle', 'square', 'triangle', 'diamond'] as const
type Marker = (typeof MARKERS)[number]

interface Style { stroke: string; dash: string; width: number; marker: Marker }

const REF_STYLE: Record<string, Style> = {
  jev: { stroke: 'var(--text)', dash: '', width: 2.5, marker: 'circle' },
  production: { stroke: 'var(--text)', dash: '7 4', width: 2, marker: 'square' },
  random: { stroke: 'var(--muted)', dash: '2 4', width: 2, marker: 'triangle' },
  oracle: { stroke: 'var(--muted)', dash: '1 5', width: 1.5, marker: 'diamond' },
}

const mean = (r: Row) => r.mean_kept_mass ?? -Infinity

/** One line per model family — its best run among `rows` — capped at TOP_N with the reference families always kept. */
function series(rows: Row[]): Row[] {
  const best = new Map<string, Row>()
  for (const r of rows) {
    if (r.kept_mass === null || r.mean_kept_mass === null) continue
    const b = best.get(r.family)
    if (!b || mean(b) < mean(r)) best.set(r.family, r)
  }
  const all = [...best.values()].sort((a, b) => mean(b) - mean(a))
  const refs = all.filter((r) => REFS.has(r.family))
  const rest = all.filter((r) => !REFS.has(r.family)).slice(0, Math.max(0, TOP_N - refs.length))
  return all.filter((r) => refs.includes(r) || rest.includes(r))
}

/** Styles are assigned in rank order of the plotted series, so the ≤7 non-reference lines never share a colour. */
function chartStyles(S: Row[]): Map<string, Style> {
  const styles = new Map<string, Style>()
  S.filter((r) => !(r.family in REF_STYLE)).forEach((r, i) =>
    styles.set(r.family, { stroke: PALETTE[i % PALETTE.length], dash: i >= PALETTE.length ? '9 3' : '', width: 1.8, marker: MARKERS[i % MARKERS.length] }),
  )
  return styles
}

function Mark({ shape, cx, cy, fill, r = 3.5 }: { shape: Marker; cx: number; cy: number; fill: string; r?: number }) {
  if (shape === 'square') return <rect x={cx - r} y={cy - r} width={2 * r} height={2 * r} fill={fill} />
  if (shape === 'triangle') return <polygon points={`${cx},${cy - r - 1} ${cx - r - 1},${cy + r} ${cx + r + 1},${cy + r}`} fill={fill} />
  if (shape === 'diamond') return <polygon points={`${cx},${cy - r - 1} ${cx + r + 1},${cy} ${cx},${cy + r + 1} ${cx - r - 1},${cy}`} fill={fill} />
  return <circle cx={cx} cy={cy} r={r} fill={fill} />
}

const Swatch = ({ st }: { st: Style }) => (
  <svg width="30" height="12">
    <line x1="0" x2="30" y1="6" y2="6" stroke={st.stroke} strokeWidth={st.width} strokeDasharray={st.dash || undefined} />
    <Mark shape={st.marker} cx={15} cy={6} fill={st.stroke} r={3} />
  </svg>
)

interface LineProps { rows: Row[]; ks: K[]; height: number; hidden: Set<string>; onToggle: (family: string) => void }

export function LineChart({ rows, ks, height, hidden, onToggle }: LineProps) {
  const S = series(rows)
  if (S.length === 0) return <p className="hint">No rows match the filters.</p>
  const styles = chartStyles(S)
  const styleOf = (family: string) => REF_STYLE[family] ?? styles.get(family)!
  const W = 900, H = height, L = 48, R = 36, T = 12, B = 30
  const kept = (r: Row, k: K) => r.kept_mass?.[k] ?? 0
  const ys = S.flatMap((r) => ks.map((k) => kept(r, k)))
  const y0 = Math.max(0, Math.floor(Math.min(...ys) * 10) / 10 - 0.05), y1 = 1
  const x = (i: number) => L + (i * (W - L - R)) / (ks.length - 1)
  const y = (v: number) => T + (H - T - B) * (1 - (v - y0) / (y1 - y0))
  const ticks: number[] = []
  for (let v = Math.ceil(y0 * 10) / 10; v <= y1 + 1e-9; v += 0.1) ticks.push(v)
  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="none" style={{ height: H }}>
        {ticks.map((v) => (
          <g key={v}>
            <line className="gridline" x1={L} x2={W - R} y1={y(v)} y2={y(v)} />
            <text x={L - 8} y={y(v) + 4} textAnchor="end">{v.toFixed(1)}</text>
          </g>
        ))}
        {ks.map((k, i) => <text key={k} x={x(i)} y={H - 8} textAnchor="middle">k={k}</text>)}
        {S.filter((r) => !hidden.has(r.family)).map((r) => {
          const st = styleOf(r.family)
          return (
            <g key={r.family}>
              <polyline fill="none" stroke={st.stroke} strokeWidth={st.width} strokeDasharray={st.dash || undefined} points={ks.map((k, i) => `${x(i)},${y(kept(r, k))}`).join(' ')}>
                <title>{r.label} · {r.serving}</title>
              </polyline>
              {ks.map((k, i) => (
                <g key={k}><Mark shape={st.marker} cx={x(i)} cy={y(kept(r, k))} fill={st.stroke} /><title>{r.label} @{k}: {kept(r, k).toFixed(3)}</title></g>
              ))}
            </g>
          )
        })}
        <line className="axis" x1={L} x2={L} y1={T} y2={H - B} />
        <line className="axis" x1={L} x2={W - R} y1={H - B} y2={H - B} />
      </svg>
      <div className="legend">
        {S.map((r) => (
          <span key={r.family} className={hidden.has(r.family) ? 'off' : ''} onClick={() => onToggle(r.family)}>
            <Swatch st={styleOf(r.family)} />{r.family} <span className="muted">{mean(r).toFixed(3)}</span>
          </span>
        ))}
      </div>
    </>
  )
}

interface BarProps { rows: Row[]; value: (r: Row) => number | null | undefined; digits: number }

/** Lowest value per family, as horizontal bars; Jev's bar is the reference. */
export function HBars({ rows, value, digits }: BarProps) {
  const best = new Map<string, { v: number; r: Row }>()
  for (const r of rows) {
    const v = value(r)
    if (v == null || v === 0) continue
    const b = best.get(r.family)
    if (!b || v < b.v) best.set(r.family, { v, r })
  }
  const arr = [...best.values()].sort((a, b) => a.v - b.v)
  if (arr.length === 0) return <p className="hint">No rows match the filters.</p>
  const max = Math.max(...arr.map((a) => a.v))
  const jev = arr.find((a) => a.r.family === 'jev')
  return (
    <div className="hbar">
      {arr.map((a) => (
        <div key={a.r.family} className={a === jev ? 'jev' : ''} style={{ display: 'contents' }}>
          <span className="lbl" title={`${a.r.label} · ${a.r.serving} · ${a.r.gpu}`}>{a.r.family} <span className="muted">{a.r.gpu}</span></span>
          <span className="t"><i style={{ width: `${(100 * a.v) / max}%` }} /></span>
          <span className={`n ${jev && a !== jev && a.v < jev.v ? 'good' : ''}`}>{a.v.toFixed(digits)}</span>
        </div>
      ))}
    </div>
  )
}

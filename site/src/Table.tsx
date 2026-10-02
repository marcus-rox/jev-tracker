import { useMemo } from 'react'
import { blob, type Row } from './types'

export interface Col {
  head: string
  value: (r: Row) => number | null
  fmt: (v: number) => string
  source: (r: Row) => string | null
}

export type Sort = { key: string; dir: 1 | -1 } | null

const LABELS: [string, (r: Row) => string][] = [
  ['ranker', (r) => r.label],
  ['serving', (r) => r.serving],
  ['runtime', (r) => r.runtime],
  ['GPU', (r) => r.gpu],
]

function Num({ row, col }: { row: Row; col: Col }) {
  const v = col.value(row)
  if (v === null) return <td className="n dim">{row.blank}</td>
  const src = col.source(row)
  return <td className="n">{src ? <a href={blob(src)} title={src} target="_blank" rel="noreferrer">{col.fmt(v)}</a> : col.fmt(v)}</td>
}

interface Props { rows: Row[]; cols: Col[]; sort: Sort; setSort: (s: Sort) => void }

export default function Table({ rows, cols, sort, setSort }: Props) {
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
  const arrow = (key: string) => (sort?.key === key ? <span className="dir">{sort.dir === 1 ? '▲' : '▼'}</span> : null)
  return (
    <>
      <p className="hint">Click a column header to sort; click again to reverse, a third time to clear. Click a number for its source JSON.</p>
      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              {LABELS.map(([h]) => <th key={h} onClick={() => click(h)}>{h}{arrow(h)}</th>)}
              {cols.map((c) => <th key={c.head} className="n" onClick={() => click(c.head)}>{c.head}{arrow(c.head)}</th>)}
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => (
              <tr key={`${r.experiment}/${r.reranker}`} className={r.highlight ? 'hl' : ''}>
                {LABELS.map(([h, get], j) => (
                  <td key={h} title={r.experiment}>{j === 0 ? <a href={blob(r.sources.config)} title={r.experiment} target="_blank" rel="noreferrer">{get(r)}</a> : get(r)}</td>
                ))}
                {cols.map((c) => <Num key={c.head} row={r} col={c} />)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

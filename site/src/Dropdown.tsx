import { useEffect, useRef, useState } from 'react'

interface Props {
  name: string
  options: string[]
  selected: Set<string>
  onChange: (selected: Set<string>) => void
}

export default function Dropdown({ name, options, selected, onChange }: Props) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('click', close)
    return () => document.removeEventListener('click', close)
  }, [open])
  const needle = query.trim().toLowerCase()
  const shown = needle ? options.filter((o) => o.toLowerCase().includes(needle)) : options
  const toggle = (o: string) => {
    const next = new Set(selected)
    if (!next.delete(o)) next.add(o)
    onChange(next)
  }
  const selectShown = () => onChange(new Set([...selected, ...shown]))
  const clearShown = () => onChange(new Set([...selected].filter((o) => !shown.includes(o))))
  return (
    <div className={open ? 'dd open' : 'dd'} ref={ref}>
      <button type="button" onClick={() => setOpen(!open)}>
        <span>{name} {selected.size === options.length ? <span className="muted">all {options.length}</span> : <span className="cnt">{selected.size} of {options.length}</span>}</span>
        <span>▾</span>
      </button>
      <div className="menu">
        {open && (
          <input className="search" type="search" placeholder={`search ${options.length} ${name.toLowerCase()}s`} value={query}
            autoFocus onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === 'Escape') setOpen(false) }} />
        )}
        <div className="all">
          <a onClick={selectShown}>select {needle ? `${shown.length} shown` : 'all'}</a>
          <a onClick={clearShown}>clear{needle ? ' shown' : ''}</a>
        </div>
        {shown.map((o) => (
          <label key={o} title={o}><input type="checkbox" checked={selected.has(o)} onChange={() => toggle(o)} />{o}</label>
        ))}
        {shown.length === 0 && <p className="none">no match</p>}
      </div>
    </div>
  )
}

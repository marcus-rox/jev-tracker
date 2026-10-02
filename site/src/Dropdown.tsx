import { useEffect, useRef, useState } from 'react'

interface Props {
  name: string
  options: string[]
  selected: Set<string>
  onChange: (selected: Set<string>) => void
}

export default function Dropdown({ name, options, selected, onChange }: Props) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('click', close)
    return () => document.removeEventListener('click', close)
  }, [open])
  const toggle = (o: string) => {
    const next = new Set(selected)
    if (!next.delete(o)) next.add(o)
    onChange(next)
  }
  return (
    <div className={open ? 'dd open' : 'dd'} ref={ref}>
      <button type="button" onClick={() => setOpen(!open)}>
        <span>{name} {selected.size === options.length ? <span className="muted">all {options.length}</span> : <span className="cnt">{selected.size} of {options.length}</span>}</span>
        <span>▾</span>
      </button>
      <div className="menu">
        <div className="all"><a onClick={() => onChange(new Set(options))}>select all</a><a onClick={() => onChange(new Set())}>clear</a></div>
        {options.map((o) => (
          <label key={o} title={o}><input type="checkbox" checked={selected.has(o)} onChange={() => toggle(o)} />{o}</label>
        ))}
      </div>
    </div>
  )
}

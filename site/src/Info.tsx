import { useEffect, useRef, useState, type CSSProperties, type MouseEvent } from 'react'
import { createPortal } from 'react-dom'

const TIP_WIDTH_PX = 320
const GAP_PX = 6
/** Below this fraction of the viewport height the tip opens upwards, so it stays on screen. */
const FLIP_AT = 0.6

/** One tooltip line, or a line with its nested sub-lines. */
export type Bullet = string | [string, string[]]

const flat = (points: Bullet[]): string => points.map((b) => (typeof b === 'string' ? b : `${b[0]} ${b[1].join(' ')}`)).join(' ')

/** A circled "i"; hover or focus shows `points` as a nested list. Rendered in a portal so table scroll boxes can't clip it. */
export default function Info({ points }: { points: Bullet[] }) {
  const [at, setAt] = useState<CSSProperties | null>(null)
  const shownAt = useRef<DOMRect | null>(null)
  const show = (el: Element) => {
    const r = el.getBoundingClientRect()
    shownAt.current = r
    const left = Math.max(GAP_PX, Math.min(r.left, document.documentElement.clientWidth - TIP_WIDTH_PX - GAP_PX))
    setAt(r.bottom > window.innerHeight * FLIP_AT ? { left, bottom: window.innerHeight - r.top + GAP_PX } : { left, top: r.bottom + GAP_PX })
  }
  const hide = () => setAt(null)
  const icon = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    if (!at) return
    // A scroll that moves the icon would leave the fixed-position tip floating; ignore any other scroll.
    const onScroll = () => {
      const now = icon.current?.getBoundingClientRect()
      if (!now || now.top !== shownAt.current?.top || now.left !== shownAt.current?.left) hide()
    }
    window.addEventListener('scroll', onScroll, true)
    return () => window.removeEventListener('scroll', onScroll, true)
  }, [at])
  return (
    <span
      ref={icon}
      className="info"
      tabIndex={0}
      role="img"
      aria-label={flat(points)}
      onMouseEnter={(e) => show(e.currentTarget)}
      onMouseLeave={hide}
      onFocus={(e) => show(e.currentTarget)}
      onBlur={hide}
      onClick={(e: MouseEvent) => e.stopPropagation()}
    >
      <svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
        <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <circle cx="8" cy="4.6" r="1.1" fill="currentColor" />
        <path d="M6.6 6.9h2.3v4.6h1v1.2H6.4v-1.2h1V8.1h-.8z" fill="currentColor" />
      </svg>
      {at && createPortal(<span className="info-tip" role="tooltip" style={{ ...at, width: TIP_WIDTH_PX }}>
          <ul>{points.map((b) => (typeof b === 'string'
            ? <li key={b}>{b}</li>
            : <li key={b[0]}>{b[0]}<ul>{b[1].map((sub) => <li key={sub}>{sub}</li>)}</ul></li>))}</ul>
        </span>, document.body)}
    </span>
  )
}

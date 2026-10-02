import { useEffect, useState } from 'react'

const THEMES = ['system', 'light', 'dark'] as const
type Theme = (typeof THEMES)[number]
const STORAGE_KEY = 'theme'

const isTheme = (v: string | null): v is Theme => THEMES.some((t) => t === v)
const stored = (): Theme => {
  const v = localStorage.getItem(STORAGE_KEY)
  return isTheme(v) ? v : 'system'
}

/** `system` leaves `data-theme` unset so the CSS `prefers-color-scheme` rule decides. */
function apply(theme: Theme) {
  if (theme === 'system') delete document.documentElement.dataset.theme
  else document.documentElement.dataset.theme = theme
}

export default function ThemeSelect() {
  const [theme, setTheme] = useState<Theme>(stored)
  useEffect(() => {
    apply(theme)
    localStorage.setItem(STORAGE_KEY, theme)
  }, [theme])
  return (
    <select className="theme" title="theme" value={theme} onChange={(e) => setTheme(e.target.value as Theme)}>
      {THEMES.map((t) => <option key={t} value={t}>{t}</option>)}
    </select>
  )
}

/** When the automation fires (its RRULE: FREQ=HOURLY;BYMINUTE=17). Keep in sync with the automation. */
export const RUN = { minute: 17, timeZone: 'America/Los_Angeles' } as const

const MINUTE_MS = 60_000
const HOUR_MS = 60 * MINUTE_MS
const DAY_MS = 24 * HOUR_MS

const WHEN_FORMAT = new Intl.DateTimeFormat('en-US', {
  timeZone: RUN.timeZone, weekday: 'short', hour: 'numeric', minute: '2-digit', timeZoneName: 'short',
})

/** The next :17 after `now`; an hourly rule at a fixed minute needs no time-zone arithmetic. */
export function nextRun(now: Date): Date {
  const at = new Date(now)
  at.setUTCMinutes(RUN.minute, 0, 0)
  if (at <= now) at.setUTCHours(at.getUTCHours() + 1)
  return at
}

export function untilText(from: Date, to: Date): string {
  const ms = Math.max(0, to.getTime() - from.getTime())
  if (ms < HOUR_MS) return `${Math.max(1, Math.round(ms / MINUTE_MS))} min`
  if (ms < DAY_MS) {
    const hours = Math.floor(ms / HOUR_MS)
    const minutes = Math.round((ms - hours * HOUR_MS) / MINUTE_MS)
    return minutes ? `${hours} h ${minutes} min` : `${hours} h`
  }
  return `${Math.floor(ms / DAY_MS)} d ${Math.round((ms % DAY_MS) / HOUR_MS)} h`
}

export const whenText = (at: Date) => WHEN_FORMAT.format(at)

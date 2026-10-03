/** When the automation runs (its RRULE: FREQ=DAILY;BYHOUR=0,6,12,18;BYMINUTE=17). Keep in sync with the automation. */
export const RUN = { hours: [0, 6, 12, 18], minute: 17, timeZone: 'America/Los_Angeles' } as const

const MINUTE_MS = 60_000
const HOUR_MS = 60 * MINUTE_MS
const DAY_MS = 24 * HOUR_MS
const MAX_RUN_SEARCH_HOURS = 8

const WHEN_FORMAT = new Intl.DateTimeFormat('en-US', {
  timeZone: RUN.timeZone, weekday: 'short', hour: 'numeric', minute: '2-digit', timeZoneName: 'short',
})

const RUN_HOUR_FORMAT = new Intl.DateTimeFormat('en-US', {
  timeZone: RUN.timeZone, hourCycle: 'h23', hour: 'numeric',
})

/** The next scheduled :17 after `now` in the Pacific time zone. */
export function nextRun(now: Date): Date {
  const at = new Date(now)
  at.setUTCMinutes(RUN.minute, 0, 0)
  if (at <= now) at.setUTCHours(at.getUTCHours() + 1)

  for (let steps = 0; steps <= MAX_RUN_SEARCH_HOURS; steps += 1) {
    const hour = Number(RUN_HOUR_FORMAT.format(at))
    if (RUN.hours.some((runHour) => runHour === hour)) return at
    at.setUTCHours(at.getUTCHours() + 1)
  }

  throw new Error(`No scheduled run found within ${MAX_RUN_SEARCH_HOURS} hours of ${now.toISOString()}.`)
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

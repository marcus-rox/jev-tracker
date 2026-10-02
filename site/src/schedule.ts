/** When the daily automation fires (its RRULE: FREQ=DAILY;BYHOUR=6;BYMINUTE=17 in America/Los_Angeles). Keep in sync with the automation. */
export const DAILY_RUN = { hour: 6, minute: 17, timeZone: 'America/Los_Angeles' } as const

const MINUTE_MS = 60_000
const HOUR_MS = 60 * MINUTE_MS
const DAY_MS = 24 * HOUR_MS

type Wall = { year: number; month: number; day: number; hour: number; minute: number; second: number }

const WALL_FORMAT = new Intl.DateTimeFormat('en-US', {
  timeZone: DAILY_RUN.timeZone, hourCycle: 'h23',
  year: 'numeric', month: 'numeric', day: 'numeric', hour: 'numeric', minute: 'numeric', second: 'numeric',
})

const WHEN_FORMAT = new Intl.DateTimeFormat('en-US', {
  timeZone: DAILY_RUN.timeZone, weekday: 'short', hour: 'numeric', minute: '2-digit', timeZoneName: 'short',
})

function wall(at: Date): Wall {
  const parts = WALL_FORMAT.formatToParts(at)
  const read = (type: Intl.DateTimeFormatPartTypes) => Number(parts.find((p) => p.type === type)?.value)
  return { year: read('year'), month: read('month'), day: read('day'), hour: read('hour'), minute: read('minute'), second: read('second') }
}

const asUtcMs = (w: Wall) => Date.UTC(w.year, w.month - 1, w.day, w.hour, w.minute, w.second)

/** UTC instant of a wall-clock time in the run's zone; the second pass absorbs a DST change between the two guesses. */
function instantOf(year: number, month: number, day: number): Date {
  const target = Date.UTC(year, month - 1, day, DAILY_RUN.hour, DAILY_RUN.minute)
  let instant = target
  for (let pass = 0; pass < 2; pass++) instant = target - (asUtcMs(wall(new Date(instant))) - instant)
  return new Date(instant)
}

export function nextRun(now: Date): Date {
  const today = wall(now)
  const todayRun = instantOf(today.year, today.month, today.day)
  return todayRun > now ? todayRun : instantOf(today.year, today.month, today.day + 1)
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

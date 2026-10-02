export type K = '50' | '100' | '150' | '200'

export interface Cost {
  warm_gpu_s: number | null
  load_s: number | null
  warm_usd: number
  in_function_usd: number
  usd_per_query: number
  usd_per_1k: number
  peak_gb: number | null
}

export interface Latency {
  run_s: number
  s_per_query: number
  h_per_1k: number
}

export interface Row {
  experiment: string
  reranker: string
  label: string
  family: string
  serving: string
  gpu: string
  buffer: string
  queries: number
  tables: string[]
  highlight: boolean
  blank: string
  kept_mass: Record<K, number> | null
  mean_kept_mass: number | null
  cost: Cost | null
  latency: Latency | null
  sources: { kept_mass: string | null; cost: string; latency: string; config: string }
}

export interface SiteData {
  ks: K[]
  experiments: Record<string, string>
  rows: Row[]
}

export const REPO = 'https://github.com/marcus-rox/jev-tracker'
export const blob = (path: string) => `${REPO}/blob/main/${path}`

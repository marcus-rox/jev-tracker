import { barsFor, clock, rateText, type ProgressState, type RerankerBar } from './progress'

/** One tqdm bar drawn as a track: `done/total [elapsed<eta, rate]`, per reranker of the running experiment. */
function Bar({ bar }: { bar: RerankerBar }) {
  const pct = bar.total > 0 ? Math.min(100, (100 * bar.done) / bar.total) : 0
  const finished = bar.total > 0 && bar.done >= bar.total
  return (
    <li className={`prog${finished ? ' done' : ''}`}>
      <div className="row">
        <span className="who">{bar.reranker}</span>
        <span className="pct">{finished ? 'done' : `${Math.floor(pct)}%`}</span>
      </div>
      <div className="track"><i style={{ width: `${pct}%` }} /></div>
      <div className="row stats">
        <span>{bar.done.toLocaleString()}/{bar.total.toLocaleString()} req{bar.shards > 1 ? ` · ${bar.shardsDone}/${bar.shards} shards` : ''}</span>
        <span>[{clock(bar.elapsedSeconds)}{finished ? '' : `<${bar.etaSeconds === null ? '?' : clock(bar.etaSeconds)}`}, {rateText(bar.ratePerSecond)}]</span>
      </div>
    </li>
  )
}

/** What the running card shows under its title: the Modal bars once a shard has reported, a sweep until then. */
export function RunProgress({ config, state }: { config: string; state: ProgressState }) {
  const bars = state.kind === 'ok' ? barsFor(state.feed, config) : []
  if (bars.length === 0) {
    return (
      <>
        <div className="bar"><i /></div>
        <div className="meta">{state.kind === 'error' ? `progress feed unavailable: ${state.message}` : 'waiting for the first Modal shard to report…'}</div>
      </>
    )
  }
  return <ul className="progress">{bars.map((bar) => <Bar key={bar.reranker} bar={bar} />)}</ul>
}

# jev-tracker

Leaderboard of Jev alternatives on Rox's frozen 75-case reranking benchmark. The data lives in
this repository; a daily automation reruns new models on Modal and opens a PR with the numbers.

- `docs/SPEC.md` — requirements R-1..R-9 and their conformance status
- `docs/PLAN.md` — phases, decisions, next steps
- `docs/PRIOR_WORK.md` — prior-work survey for the crawler (R-8): sources, rate limits, what to copy
- `docs/AUTOMATION.md` — the daily automation's runbook (R-9): crawl, triage, run, regenerate, open a PR
- `jev_tracker/` — harness: System One contract, methods, kept-mass metric, Modal runner,
  experiment lifecycle (`run` / `status` / `finish` / `costs` / `latency`)
- `configs/` — one YAML per experiment (which models, methods, GPUs, shards)
- `data/benchmark/` — the 75 cases and labels (frozen)
- `data/jev/` — Jev's committed answers
- `data/experiments/<id>/` — config, calls, raw answers (gzipped), kept-mass, costs, latency
- `data/registry.yaml` — row labels (model family, serving, GPU) for every reranker shown
- `site/` — Vite + React static site: summary / quality / cost / latency tables, filters, compare,
  "Evaluate a new model" form; `site/public/data/rows.json` is generated, `site/dist/` is the build
- `crawler/` — R-8: finds new Jev / Kev / Laya / decision-model mentions (GitHub, Hugging Face,
  arXiv, web); `crawler/queries.yaml`, `crawler/seen.jsonl`, `crawler/candidates/<date>.jsonl`

## Run

```bash
uv sync
export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...   # Modal workspace rox-research
uv run python -m jev_tracker.experiment run configs/kev4b_smoke.yaml   # 2 cases, ~$0.05
uv run python -m jev_tracker.experiment status <id>
uv run python -m jev_tracker.experiment finish <id>                    # pull raws, score, cost, latency
uv run pytest                                                          # offline tests
```

## Site

```bash
uv run python -m jev_tracker.site_data     # registry.yaml + data/experiments -> site/public/data/rows.json
cd site && npm ci && npm run build          # -> site/dist (committed)
python3 -m http.server -d site/dist 8000    # http://localhost:8000
```

Every number on the page links to the JSON it came from. "Evaluate a new model" writes the
experiment YAML and opens a prefilled GitHub issue labelled `evaluate`; nothing is sent from the
browser.

## Crawler

```bash
uv run python -m crawler                                  # since the last run (else 7 days)
uv run python -m crawler --since 2026-09-01 --out crawler/candidates/
GITHUB_TOKEN=... uv run python -m crawler                  # 30 instead of 10 GitHub searches/min
```

Queries are in `crawler/queries.yaml` (one list per source). Each run dedupes on URL against
`crawler/seen.jsonl`, appends the new URLs there and writes one JSON object per candidate
(`source, url, title, snippet, first_seen, query`) to `crawler/candidates/<YYYY-MM-DD>.jsonl`.
A failing (source, query) is printed and skipped; the exit code is 1 if any failed. DuckDuckGo
answers with a bot challenge from some networks; that source then logs and returns nothing.
Prior-work survey: `docs/PRIOR_WORK.md`.

## Automation

A Devin Automation runs `docs/AUTOMATION.md` once a day and opens a PR; nothing lands on `main`
without a merge. Devin's decisions are data in that PR: `crawler/triage/<date>.yaml` (a verdict
per candidate) and the `configs/*.yaml` it wrote for runnable ones.

```bash
uv run python -m crawler                                   # 1. new candidates
uv run python -m crawler.triage check crawler/triage/<date>.yaml crawler/candidates/<date>.jsonl
uv run python -m jev_tracker.evaluate_issues               # 2. open `evaluate` issues -> configs/
uv run python -m jev_tracker.experiment run configs/<new>.yaml --wait   # 3. per new config
uv run python -m jev_tracker.site_data && (cd site && npm ci && npm run build)   # 4. regenerate
```

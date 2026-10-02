# jev-tracker

Leaderboard of Jev alternatives on Rox's frozen 75-case reranking benchmark. The data lives in
this repository; a daily automation reruns new models on Modal and opens a PR with the numbers.

- `docs/SPEC.md` — requirements R-1..R-9 and their conformance status
- `docs/PLAN.md` — phases, decisions, next steps
- `jev_tracker/` — harness: System One contract, methods, kept-mass metric, Modal runner,
  experiment lifecycle (`run` / `status` / `finish` / `costs` / `latency`)
- `configs/` — one YAML per experiment (which models, methods, GPUs, shards)
- `data/benchmark/` — the 75 cases and labels (frozen)
- `data/jev/` — Jev's committed answers
- `data/experiments/<id>/` — config, calls, raw answers (gzipped), kept-mass, costs, latency
- `data/registry.yaml` — row labels (model family, serving, GPU) for every reranker shown
- `site/` — Vite + React static site: summary / quality / cost / latency tables, filters, compare,
  "Evaluate a new model" form; `site/public/data/rows.json` is generated, `site/dist/` is the build

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

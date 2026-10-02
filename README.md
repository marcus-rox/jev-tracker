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
Prior-work survey: `docs/CRAWLER_PRIOR_WORK.md`.

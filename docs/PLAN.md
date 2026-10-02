# jev-tracker — a leaderboard of Jev alternatives on Rox's frozen 75-case benchmark

**Status · 2026-10-02 · Phases 1–2 merged (PR #1, #2); Phase 3 crawler merged (PR #3, #4); Phase 3b
(twitter/hackernews/Tavily sources, `hf:<id>@<sha>` key, evaluate issues → configs, triage file, runbook)
in one PR; the daily Devin automation exists, disabled until that PR merges.** Harness ported from
`rox-research/projects/reranker_alts`; Kev-4B, Laya and batched Kev-27B (pinned to the 2026-09-24
weights) reran on Modal within 0.002 kept-mass of rox-research at every k.

## The objective

Marcus can open one local page, see the quality / cost / latency tables of every model run on the
75 cases, filter runs, and submit a new URL or Hugging Face model; a daily automation
finds new Jev alternatives, runs them, and opens a PR with the numbers. Success for Phase 1: the
three validation reruns (R-4) match rox-research within 0.002 kept-mass at every k.

## Constraints

- Benchmark runs use all 75 cases; smaller N is a smoke test only (Marcus, Laya session).
- Benchmark runs are batched; one request at a time only for smoke tests (Marcus, Laya session).
- tqdm progress for every download and experiment wait (Marcus, standing).
- Raw answers live in the repo under `data/` (Marcus, 2026-10-01). GitHub warns at 50 MB per
  file: raw files are gzipped (~1–2 MB each).
- Automation never commits to `main`; it opens a PR (Marcus, 2026-10-01).
- Secrets never enter the repo: Modal tokens from `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`, API
  keys from Modal Secrets named in the config, the GitHub PAT from Devin's secret store.

## Previous steps

| Step | Outcome | What it established |
|---|---|---|
| Access check with the PAT | ✅ | Push, workflows, actions, PRs work; Pages has no site yet |
| Context: Slack threads, Jev sessions, report, source branch | ✅ | The seam to port is producers → `RawRecord` → evaluator |
| Port + validate (Phase 1, PR #1) | ✅ | Kev-4B, Laya, Kev-27B@2026-09-24 within 0.002 at every k; Kev-27B v2 weights differ by ~0.01 |
| Site (Phase 2) | ✅ | One `rows.json` from `registry.yaml` + experiments reproduces the report's cells; static Vite build |

## Next steps

1. **Daily automation (Phase 3b)**: R-8 edits (twitter, hackernews, Tavily web, `hf:<id>@<sha>`
   key), `jev_tracker.evaluate_issues`, `crawler/triage.py`, `docs/AUTOMATION.md`, then the Devin
   automation that runs the runbook daily and opens a PR. Falsifier: the first automation PR has no
   new candidate or fails to regenerate the site.

## Decided

- **Package name `jev_tracker`, paths relative to the repo root** (not `projects/...`): the repo
  has one project; commands run from anywhere.
- **Modal app `jev-tracker`, Volume `jev-tracker-runs`, HF cache Volume `rox-research--hf-cache`
  reused** (same workspace; Kev/Laya weights already cached). Auth from env vars, no
  `env_var_management` layer.
- **Sources kept: production, answers, kev, laya, api.** `api` generalises the Liquid d1 producer
  (URL + model + Modal secret + key env var) and is what "put in a URL" on the site maps to.
- **Dropped code: vLLM/SGLang/llama.cpp engines, FP8, CUDA-graph bank, CLM producer, parity/drift.**
  Closed hypotheses; the experiments they produced are carried as data with their registry rows.
- **Raw answers gzipped** (`raw_<reranker>_<id>.json.gz`); the loader accepts both suffixes so
  rox-research files port unchanged.
- **Row labels in `data/registry.yaml`** instead of the hard-coded `Row(...)` lists of
  `report_tables.py`: the site needs them as data, and a new experiment adds rows without code.
- **Experiment ids stay `YYYY_MM_DD_HH_MM_SS_<petname>`**: Marcus's convention; the 18 carried
  experiments keep their ids so the report's references resolve.
- **Kev-27B revalidation is the batched config only** (Marcus): the sequential and FP8 runs are
  not rerun.
- **Pin the Hub revision when a model's `main` moves.** `jaredpalmer/kev-27b` got new weights on
  2026-09-30 ("Kev-27B v2", round 23) after the rox-research runs (2026-09-24 weights, revision
  `01b8199`). The unpinned rerun (`2026_10_01_23_53_25_fit-macaw`) is ~0.01 kept-mass lower at
  k ≤ 150 and is kept as the v2 row; R-4 is checked against the pinned rerun
  (`configs/kev27b_batched_v1.yaml`, `model: jaredpalmer/kev-27b@01b8199…`). Kev-4B's weights last
  changed 2026-09-24, before its reference run, so it needed no pin.

## Out of scope

- Slack reporting of the daily run (deferred by Marcus).
- Render / Supabase: later; the static `site/` folder is what moves.
- Re-running CLM or any Kev-27B serving hypothesis.
- Jev via its API (needs `JEV_API_KEY`).

## Open

- ~~Whether Kev-4B's and Laya's own rerun spread is within 0.002~~ — yes: both reran within 0.002
  at every k (`tests/test_validation.py`).
- ~~Devin automation configuration~~ — a Devin Automation with a daily schedule trigger, running as
  Marcus (his PAT and Modal tokens), following `docs/AUTOMATION.md`; no Slack.

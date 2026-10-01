# jev-tracker — specification

**Status · 2026-10-01 · 9 requirements, 9 unbound (nothing built yet).** Approved by Marcus on
2026-10-01 with four decisions (raw answers in the repo under `data/`; automation opens a PR; one
batched Kev-27B revalidation; no Slack report). Only Marcus edits this file after that.

## Requirement levels

```
R-1   Any decision model can be scored on the frozen 75-case set           jev_tracker.experiment
├── R-2    Every scorer speaks the System One contract                    jev_tracker.systemone, methods
├── R-3    An experiment is fully recorded in the repository              jev_tracker.experiment
└── R-4    The ported harness reproduces the committed results            tests/, data/experiments

R-5   The three tables are one static website with filters               site/
├── R-6    The site is generated from the committed experiments           jev_tracker.site_data
└── R-7    A new model can be submitted from the site                     site/, .github

R-8   New Jev mentions and open-source alternatives are found daily      crawler/
└── R-9    A daily automation turns candidates into a PR                  automation (Devin)
```

| Module | Level 1 | Level 2 |
|---|---|---|
| `jev_tracker.experiment`, `modal_app` | R-1 | R-3 |
| `jev_tracker.systemone`, `methods`, `rerankers`, `kept_mass` | | R-2 |
| `tests/`, `data/experiments` | | R-4 |
| `site/` | R-5 | R-7 |
| `jev_tracker.site_data` | | R-6 |
| `crawler/` | R-8 | |
| Devin automation | | R-9 |

## Purpose

An internal leaderboard of decision models ("Jev alternatives") on Rox's frozen 75-query
parent-child reranking benchmark. Every model is asked the same System One questions, its raw
answers are committed, and kept-mass@k, cost and latency are derived from those files. A static
website shows the quality, cost and latency tables with filters; new models arrive either from a
form on the site or from a crawler, and a daily Devin automation runs them on Modal and opens a PR
with the new data and the regenerated site.

## Requirements

- **R-1 Scoring.** `experiment run <config.yaml>` SHALL score every model source named in the
  config on the 75 cases (or the first N for a smoke test) on Modal, sharded over GPUs, and SHALL
  show tqdm progress for downloads, shard waits and pulls. Supported sources: `production`,
  `answers`, `kev` (any Hugging Face Kev checkpoint, Kev's own server, batched), `laya` (the three
  official checkpoints, batched forward passes), and `api` (any hosted model that answers the
  System One request at a URL, key from a named Modal secret).
  - Scenario: `configs/kev4b_vs_jev.yaml` → one experiment directory with Kev-4B's answers on all
    75 cases and a kept-mass table next to production and Jev.
  - Scenario: an `api` source with LiquidAI's URL and `d1:free` reproduces the d1 run shape (HTTP
    retries on 429/5xx, batch split on "input limit").
- **R-2 Contract.** Every scorer SHALL be asked through `SystemOneRequest` and answered through
  `SystemOneResponse`; stored answers SHALL be `RawRecord`s whose `scores` bind each child to its
  answer. Nothing downstream of `record` can tell which model answered.
  - Scenario: Jev's committed answers and a Kev run validate through the same types and give the
    evaluator the same keys per case.
- **R-3 Artifacts.** One experiment = `data/experiments/<YYYY_MM_DD_HH_MM_SS_petname>/` holding
  the config as run, the Modal calls, every scored model's raw answers (gzipped JSON), and the
  kept-mass, costs and latency JSON. Row labels (model family, serving, GPU, queries) live in
  `data/registry.yaml`, one entry per (experiment, reranker). Nothing is overwritten.
  - Scenario: deleting the Modal Volume loses nothing the site shows.
- **R-4 Validation.** Reruns of Kev-4B (`kev4b_vs_jev`), Laya batched (`laya_vs_jev`) and Kev-27B
  batched (`kev27b_batched`) on all 75 cases SHALL match the committed kept-mass@k of the same
  config from rox-research within |Δ| ≤ 0.002 at every k in {50, 100, 150, 200}. A prod + Jev
  evaluation of the ported metric SHALL reproduce 0.711/0.837 and 0.893/0.961 at @50/@200 exactly.
  - The 0.002 band is Kev-27B's measured run-to-run spread (rox-research H-1 reruns); if a model's
    own rerun spread is larger, the comparison reports it and Marcus decides.
- **R-5 Site.** `site/dist/` SHALL be a static build served by `python3 -m http.server` showing a
  summary table (each model's best ranker), the quality table (kept-mass@50/100/150/200), the cost
  table ($/run, $/query, $/1k) and the latency table (s/query, h/1k), with filters on model
  family, serving, GPU, experiment, queries=75, and k columns; column sort; and side-by-side
  comparison of 2+ selected experiments. Every number links to its source JSON.
- **R-6 Site data.** `python -m jev_tracker.site_data` SHALL regenerate `site/public/data/*.json`
  from `data/experiments/` and `data/registry.yaml` with no hand edits; the attached report's
  numbers SHALL appear unchanged on day one.
- **R-7 Submit.** The site SHALL have an "Evaluate a new model" form (URL or Hugging Face id, model
  name, source type, methods) that produces the YAML config and opens a prefilled GitHub issue
  labelled `evaluate` on this repository. No token is shipped to the browser.
- **R-8 Crawler.** `python -m crawler` SHALL search GitHub, Hugging Face Hub, arXiv and the web for
  new Jev / System One / decision-model mentions since the last run, dedupe against
  `crawler/seen.jsonl`, and write `crawler/candidates/<date>.jsonl` (url, title, snippet,
  first_seen). Design is preceded by a short prior-work survey in `docs/`.
- **R-9 Automation.** A daily Devin automation SHALL crawl, triage candidates (runnable now /
  needs adapter / not a Jev), run approved configs and `evaluate` issues on Modal, regenerate the
  site data, and open a PR to `main`. Python does the mechanical steps; Devin only decides what to
  include.

## Not required

- Kev-27B serving hypotheses (vLLM, SGLang, llama.cpp, FP8, CUDA-graph bank width): refuted or
  closed in rox-research; their artifacts are carried over as data only.
- CLM-8B rerun: its producer needs a vLLM encoder image; its committed run is carried as data.
- Slack reporting (Marcus: "skip for now").
- A database or hosted backend: data stays in Git until a file nears GitHub's 50 MB warning.
- Jev itself through its API (needs `JEV_API_KEY`; its committed answers are the reference).
- Authentication on the website (internal localhost use; revisit at the Render migration).

## Clarifications

None.

## Conformance

| Requirement | Module | Test | Status |
|---|---|---|---|
| R-1 | `jev_tracker.experiment` | `tests/test_experiment.py::test_R1_*` | unbound |
| R-2 | `jev_tracker.systemone` | `tests/test_contract.py::test_R2_*` | unbound |
| R-3 | `jev_tracker.experiment` | `tests/test_experiment.py::test_R3_*` | unbound |
| R-4 | `data/experiments` | `tests/test_validation.py::test_R4_*` | unbound |
| R-5 | `site/` | manual (Marcus) + `tests/test_site_data.py` | unbound |
| R-6 | `jev_tracker.site_data` | `tests/test_site_data.py::test_R6_*` | unbound |
| R-7 | `site/` | `tests/test_site_data.py::test_R7_issue_url` | unbound |
| R-8 | `crawler/` | `tests/test_crawler.py::test_R8_*` | unbound |
| R-9 | automation | manual: first PR opened by the automation | unbound |

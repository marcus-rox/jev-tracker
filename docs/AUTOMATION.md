# Daily automation runbook (R-9)

What the daily Devin session does, in order. Every step but triage is one command; triage is the
one decision Devin makes, and it is written down as data so the PR shows it.

**Schedule**: once a day. **Identity**: Marcus (his fine-grained PAT `MARCUS_SITE_GITHUB_TOKEN`
for GitHub, `MODAL_TOKEN_ID_ROX_RESEARCH` / `MODAL_TOKEN_SECRET_ROX_RESEARCH` for Modal).
**Output**: one PR `automation/<YYYY-MM-DD>` into `main`, or no PR when nothing is new.
**Budget**: at most 3 new experiments per run; a config whose GPUs or shards exceed those of
`configs/kev27b_batched.yaml` is left as `needs_adapter` with the reason "over budget".

## Steps

1. **Checkout**: clone `main` over HTTPS with the PAT (askpass helper; never print it), branch
   `automation/<date>`, `uv sync`, `export MODAL_TOKEN_ID=$MODAL_TOKEN_ID_ROX_RESEARCH
   MODAL_TOKEN_SECRET=$MODAL_TOKEN_SECRET_ROX_RESEARCH`.
2. **Crawl**: `uv run python -m crawler` → `crawler/candidates/<date>.jsonl`, `crawler/seen.jsonl`.
   Links submitted from the site (`requests/<date>/*.json`) come along as source `submitted`.
   Exit 1 means a (source, query) failed; keep going, list the failures in the PR.
3. **Triage** (the decision): read today's candidates and write `crawler/triage/<date>.yaml`, one
   entry per `key`:
   - `runnable`: a Hugging Face model that speaks System One through an existing source (`kev` for
     Kev-family checkpoints, `laya` for Laya) or a hosted endpoint with a URL + model name (`api`
     source; the key must already be a Modal Secret — otherwise `needs_adapter`). Write
     `configs/<slug>.yaml` next to `configs/kev4b_vs_jev.yaml` (same shape: prod + Jev answers +
     the new model, all 75 cases) and add the model's row labels to `data/registry.yaml`.
   - `needs_adapter`: a real Jev alternative the harness cannot call yet (new serving stack, new
     request format, key not provisioned, over budget). Say what is missing in `reason`.
   - `not_jev`: unrelated hit (a person named Kev, a repo about something else). One-line reason.
   A `submitted` candidate is a human asking for exactly that link: open it, work out what it is
   (model, repo, paper, endpoint) and give it one of the three verdicts like any other candidate.
   Then `uv run python -m crawler.triage check crawler/triage/<date>.yaml
   crawler/candidates/<date>.jsonl` must exit 0.
4. **Issues**: `GITHUB_TOKEN=<PAT> uv run python -m jev_tracker.evaluate_issues` turns open
   `evaluate` issues into `configs/issue_<n>_<slug>.yaml` (unauthenticated api.github.com is
   rate-limited from Devin VMs; the issue stays open until a human closes it, so the file-exists
   skip is what stops a rerun). Add registry rows for their rerankers as in step 3.
5. **Run**: for each config written in steps 3–4 (max 3):
   `uv run python -m jev_tracker.experiment run configs/<name>.yaml --wait`. A failed run is
   reported in the PR, not retried.
6. **TLDR** (the second decision): rewrite `data/tldr.md`, 3–5 plain sentences for someone who
   opens the site cold: who leads the benchmark and by how much, what today's runs added, what
   changed since yesterday. Numbers come from `data/experiments`; no markdown headings.
7. **Regenerate**: `uv run python -m jev_tracker.site_data` (reads `data/tldr.md`), then
   `cd site && npm ci && npm run build && cd ..`.
8. **Check**: `uv run ruff check . && uv run ruff format --check . && uv run pytest -q`.
9. **PR**: commit `crawler/`, `configs/`, `data/`, `site/public/data`, `site/dist` (explicit
   paths, no `git add .`); push; open a PR titled `Daily <date>: <n> candidates, <m> runs` whose body
   has the triage counts per verdict, one line per run with kept-mass@50/200 and $/run, the crawler
   failures if any, and `Closes #<n>` for each `evaluate` issue run. No Slack (Marcus: skip for now).
10. **Nothing new** (no candidates, no issues, no submissions): stop without a PR and say so in the session's final
   message.

## What the automation never does

- Push to `main`, force-push, amend, or edit `docs/SPEC.md`.
- Run more than 3 experiments or a config over the Kev-27B budget line.
- Print or commit a token.

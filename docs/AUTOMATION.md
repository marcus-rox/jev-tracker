# Daily automation runbook (R-9)

What the daily Devin session does, in order. Every step but triage is one command; triage is the
one decision Devin makes, and it is written down as data so the PR shows it.

**Schedule**: once a day. **Identity**: Marcus (his fine-grained PAT `MARCUS_SITE_GITHUB_TOKEN`
for GitHub, `MODAL_TOKEN_ID_ROX_RESEARCH` / `MODAL_TOKEN_SECRET_ROX_RESEARCH` for Modal).
**Output**: one PR `automation/<YYYY-MM-DD>` into `main`, or no PR when nothing is new.
**Budget**: at most 10 new experiments per run; a config whose GPUs or shards exceed those of
`configs/kev27b_batched.yaml` is left as `needs_adapter` with the reason "over budget".

## Steps

1. **Checkout**: clone `main` over HTTPS with the PAT (askpass helper; never print it), branch
   `automation/<date>`, `uv sync`, `export MODAL_TOKEN_ID=$MODAL_TOKEN_ID_ROX_RESEARCH
   MODAL_TOKEN_SECRET=$MODAL_TOKEN_SECRET_ROX_RESEARCH`.
2. **Crawl**: first the Slack queries (the `slack:` list in `crawler/queries.yaml`): for each, call
   the Slack MCP tool `slack_search_public_and_private` with `keywords=[query]`,
   `filters="after:<since YYYY-MM-DD>"`, `sort="timestamp"`, `include_context=false`, paging with
   the cursor until exhausted, and save the `results` text to `/tmp/slack/<query slug>.md`
   (`python -c 'from crawler.slack import slug; print(slug(q))'`). Then
   `uv run python -m crawler --slack-results /tmp/slack` → `crawler/candidates/<date>.jsonl`,
   `crawler/seen.jsonl`. Without the Slack MCP (or the flag) the crawler prints that it skipped
   `slack` and the other six sources run.
   Text submitted from the site (`requests/<date>/*.json`) comes along as source `submitted`.
   Exit 1 means a (source, query) failed; keep going, list the failures in the PR.
3. **Triage** (the decision): read today's candidates and write `crawler/triage/<date>.yaml`, one
   entry per `key`:
   - `runnable`: a Hugging Face model that speaks System One through an existing source (`kev` for
     Kev-family checkpoints, `laya` for Laya) or a hosted endpoint with a URL + model name (`api`
     source; the key must already be a Modal Secret — otherwise `needs_adapter`). Write
     `configs/<slug>.yaml` next to `configs/kev4b_vs_jev.yaml` (same shape: prod + Jev answers +
     the new model, all 75 cases) and add the model's row labels to `data/registry.yaml`.
   - `needs_adapter`: a real Jev alternative the harness cannot call yet (new serving stack, new
     request format, key not provisioned, over budget). Say what is missing in `reason`. When it
     looks worth the adapter work (a distinct open model family, a hosted endpoint that only needs
     a key, a Kev/Laya variant the source almost loads), also set `config: configs/<slug>.yaml`
     (the file does not exist yet): that makes it a *proposal* Marcus approves or skips on the site.
   - `not_jev`: unrelated hit (a person named Kev, a repo about something else). One-line reason.
   A `submitted` candidate is whatever a human typed into the site (usually a link): open or
   search for it, work out what it is (model, repo, paper, endpoint) and give it one of the three
   verdicts like any other candidate.
   Then `uv run python -m crawler.triage check crawler/triage/<date>.yaml
   crawler/candidates/<date>.jsonl` must exit 0.
3b. **Queue** (what the site shows while runs are going): `uv run python -m
   jev_tracker.evaluation_queue add crawler/triage/<date>.yaml` puts every runnable config in
   `data/queue.json` as `queued` and every needs_adapter decision with a `config` as `proposed`;
   then `… start configs/<a>.yaml configs/<b>.yaml` (the ≤10 that step 5 will run) marks them
   `running`. Commit `data/queue.json` alone as the branch's first
   commit and push that one commit to `main` too (`git push origin HEAD:main`; the branch is
   `main` + this commit, so it is a fast-forward). This is the one write to `main` the run makes.
4. **Issues**: `GITHUB_TOKEN=<PAT> uv run python -m jev_tracker.evaluate_issues` turns open
   `evaluate` issues into `configs/issue_<n>_<slug>.yaml` (unauthenticated api.github.com is
   rate-limited from Devin VMs; the issue stays open until a human closes it, so the file-exists
   skip is what stops a rerun). Add registry rows for their rerankers as in step 3.
4b. **Approvals**: `queued` items in `data/queue.json` whose config file does not exist are
   proposals Marcus approved on the site since the last run (the site's server commits the click
   to `main`). Approval means "build what the note says is missing": write the source / adapter /
   Modal Secret wiring and the config, then run it like any other. One that cannot be finished in
   the session stays `queued` with its note and the PR says what is left.
5. **Run**: for each config written in steps 3–4b, approved ones first (max 10):
   `uv run python -m jev_tracker.experiment run configs/<name>.yaml --wait`. A failed run is
   reported in the PR, not retried. Afterwards `uv run python -m jev_tracker.evaluation_queue
   done configs/<name>.yaml …` for every config that ran (failed ones too; the PR says why): a
   finished model leaves the queue and exists only as its rows in the site data. Configs that were
   queued but not run today stay `queued` for tomorrow.
6. **TLDR** (the second decision): rewrite `data/tldr.md` as nested markdown bullets for someone
   who opens the site cold. Two levels only: a top-level bullet is one claim, its children are the
   numbers that support it. At most 5 words per bullet. No headings, no tables, no
   prose paragraphs. Order: who leads the benchmark, the closest alternative, what today's runs
   added, what changed since yesterday. Numbers come from `data/experiments` exactly as in the
   data. Write it in ASD-STE100 (Simplified Technical English) at about 90% compliance: one idea
   per bullet, active voice, present tense, approved general words (`use` not `utilize`, `show`
   not `demonstrate`), no idioms.
7. **Regenerate**: `uv run python -m jev_tracker.site_data` (reads `data/tldr.md`), then
   `cd site && npm ci && npm run build && cd ..`.
8. **Check**: `uv run ruff check . && uv run ruff format --check . && uv run pytest -q -n auto`.
9. **PR**: commit `crawler/`, `configs/`, `data/` (including `data/queue.json` with the finished
   items removed), `site/public/data`, `site/dist` (explicit
   paths, no `git add .`); push; open a PR titled `Daily <date>: <n> candidates, <m> runs` whose body
   has the triage counts per verdict, one line per run with kept-mass@50/200 and $/run, the crawler
   failures if any, and `Closes #<n>` for each `evaluate` issue run.
9b. **Slack DM** to Marcus Dominguez-Kuhne (Slack user `U0BQQC4046P`), with the session's Slack
   tools, on every run (also when a step failed). First line: `Please merge: <PR URL>` (or
   `No PR today: nothing new`). Then one line per model run today (kept-mass@50/200, $/run), and
   the list of `proposed` items awaiting his approval on the site (label + note). No channel posts.
10. **Nothing new** (no candidates, no issues, no submissions): stop without a PR, send the step 9b
   DM saying so, and say so in the session's final message.

## What the automation never does

- Push to `main` (except the single `data/queue.json` fast-forward in step 3b), force-push, amend,
  or edit `docs/SPEC.md`.
- Run more than 10 experiments or a config over the Kev-27B budget line.
- Run a `needs_adapter` model Marcus has not approved on the site (or via
  `jev_tracker.evaluation_queue approve`).
- Print or commit a token.

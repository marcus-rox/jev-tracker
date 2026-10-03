# Hourly automation runbook (R-9)

What each hourly Devin session does. Every step but triage is one command; triage is
the one decision Devin makes, and it is written down as data so the commit shows it.

**Schedule**: once an hour (at :17). **Identity**: Marcus (his fine-grained PAT `MARCUS_SITE_GITHUB_TOKEN`
for GitHub, `MODAL_TOKEN_ID_ROX_RESEARCH` / `MODAL_TOKEN_SECRET_ROX_RESEARCH` for Modal).
**Output**: commits pushed straight to `main` (no branch, no PR — the site redeploys from each):
the queue as runs start (A1), the search results (B4), and the results (J5); nothing when nothing
is new.
**Budget**: at most 10 new experiments per run; a config whose GPUs or shards exceed those of
`configs/kev27b_batched.yaml` is left as `needs_adapter` with the reason "over budget".
**Batching** (hard rule, Marcus 2026-10-02): every experiment runs batched — never launch a
config whose rerankers answer requests one at a time. Each engine must use the harness's batched
path: server-side request queueing (`concurrency` > 1) or multi-request forward batches
(`forward_batch` > 1 bodies per predict call). A `queued` config or running experiment that is
un-batched (`concurrency: 1` with `forward_batch: 1`) gets stopped, its adapter gets the batched
path it is missing, and it re-runs batched — the commit message records the kill and the re-run.
A proposal whose adapter cannot be made batched in the session stays `queued`/`proposed` with
the gap in its note; it never runs un-batched. Launch every runnable experiment in parallel.
**Numbers** (like-for-like): a run's cost is its `costs_<id>.json` `warm_usd_per_1k` — warm
seconds (loaded server to last answer, at the run's full load: `concurrency` requests in flight
per GPU, 16 by default) summed over every GPU shard, billed at Modal GPU $/s plus reserved host
RAM $/GiB/s, failed calls $0, scaled x1000/queries. Its latency is `latency_<id>.json` `mean_s` —
per-query wall clock, first request sent to last answer back. A registry row whose numbers cannot
be derived (no per-request start times, no raw answers, a failed run) gets `deprecated: <reason>`
in `data/registry.yaml` and never reaches the site.
**Kill rule** (Marcus 2026-10-02): a run only exists to beat production — the `prod` baseline is
$63/1k queries ($4.73 per 75-query run) and a mean of 6.21 s/query (wall clock, same definition
as the run's `mean_s`). Kill an experiment when its projected $/1k AND projected mean s/query
both exceed those numbers (strictly worse buys nothing); cancel its
Modal calls, mark it `failed` ("killed: over production cost+latency line"), and record the kill
in the commit message. A run that is cheaper OR faster stays — report it.

## Steps

The run is two tracks that go at the same time, then one join:

- **Track A — run** takes items from `data/queue.json` and starts them on Modal right away.
- **Track B — search** does the web search and puts what it finds into the queue.

Track A starts first because its Modal runs take the longest. Every `experiment run --wait`
goes in its own background shell (`… > /tmp/run_<name>.log 2>&1 &`) so the session works on
Track B while the GPUs run. Only the session touches git, one commit at a time, and each commit
lists its paths explicitly (never `data/experiments/`, which the background runs are still
writing).

1. **Checkout**: clone `main` over HTTPS with the PAT (askpass helper; never print it), stay on
   `main`, `uv sync`, `export MODAL_TOKEN_ID=$MODAL_TOKEN_ID_ROX_RESEARCH
   MODAL_TOKEN_SECRET=$MODAL_TOKEN_SECRET_ROX_RESEARCH`.

### Track A — run (start at once)

A1. **Start**: pick the `queued` items in `data/queue.json` whose config file exists, approved
   ones first (max 10). `uv run python -m jev_tracker.evaluation_queue start configs/<a>.yaml …`
   marks them `running`; commit `data/queue.json` alone and push it to `main` right away
   (`git push origin HEAD:main`) so the board shows the runs. Then, for each, in the background:
   `uv run python -m jev_tracker.experiment run configs/<name>.yaml --wait`. Note the experiment
   id each one prints. A failed run is reported in the commit message, not retried.
A2. **Approvals**: `queued` items whose config file does not exist are proposals Marcus approved on
   the site since the last run (the site's server commits the click to `main`). Approval means
   "build what the note says is missing": write the source / adapter / Modal Secret wiring and the
   config, then start it as in A1 (within the same 10). One that cannot be finished in the session
   stays `queued` with its note and the commit message says what is left. Do this after B1–B4,
   so the search is not held up by adapter work.
A3. **Pick up Track B's items**: when B4 has pushed, start the newly `queued` runnable configs and
   the B3 issue configs as in A1, up to the budget of 10 for this run. The rest stay `queued` for
   the next hour.

### Track B — search (while Track A runs)

B1. **Crawl**: first the Slack queries (the `slack:` list in `crawler/queries.yaml`): for each, call
   the Slack MCP tool `slack_search_public_and_private` with `keywords=[query]`,
   `filters="after:<since YYYY-MM-DD>"`, `sort="timestamp"`, `include_context=false`, paging with
   the cursor until exhausted, and save the `results` text to `/tmp/slack/<query slug>.md`
   (`python -c 'from crawler.slack import slug; print(slug(q))'`). Then
   `uv run python -m crawler --slack-results /tmp/slack` → `crawler/candidates/<date>.jsonl`,
   `crawler/seen.jsonl`. Without the Slack MCP (or the flag) the crawler prints that it skipped
   `slack` and the other six sources run.
   Text submitted from the site (`requests/<date>/*.json`) comes along as source `submitted`.
   Exit 1 means a (source, query) failed; keep going, list the failures in the commit message.
B2. **Triage** (the decision): read today's candidates and write `crawler/triage/<date>.yaml`, one
   entry per `key`:
   - `runnable`: a Hugging Face model that speaks System One through an existing source (`kev` for
     Kev-family checkpoints, `laya` for Laya) or a hosted endpoint with a URL + model name (`api`
     source; the key must already be a Modal Secret — otherwise `needs_adapter`). Write
     `configs/<slug>.yaml` next to `configs/kev4b_vs_jev.yaml` (same shape: prod + Jev answers +
     the new model, all 75 cases) and add the model's row labels to `data/registry.yaml`, each with
     its `runtime` (`PyTorch` for weights the harness loads in-process; the allowed values are
     `RUNTIMES` in `jev_tracker/site_data.py`).
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
B3. **Issues**: `GITHUB_TOKEN=<PAT> uv run python -m jev_tracker.evaluate_issues` turns open
   `evaluate` issues into `configs/issue_<n>_<slug>.yaml` (unauthenticated api.github.com is
   rate-limited from Devin VMs; the issue stays open until a human closes it, so the file-exists
   skip is what stops a rerun). Add registry rows for their rerankers as in B2.
B4. **Queue**: `uv run python -m jev_tracker.evaluation_queue add crawler/triage/<date>.yaml`
   puts every runnable config in `data/queue.json` as `queued` and every needs_adapter decision
   with a `config` as `proposed`. Commit `crawler/`, `configs/`, `data/registry.yaml` and
   `data/queue.json` and push to `main` right away (`git pull --rebase origin main` first if it
   moved), so the board shows the new items.

### Join (when every Track A run has exited)

J1. **Verify and close** every config that ran: `uv run python -m jev_tracker.evaluation_queue
   done configs/<name>.yaml --experiment <id>` (the id `run` printed). This runs
   `jev_tracker.experiment verify <id>`, which checks the Modal side — every spawned call
   finished, every scored reranker answered every case, the kept-mass table exists — and removes
   the item only when all of that holds; otherwise the item becomes `failed` on the board with the
   evidence (call states, cases answered per reranker) and stays until Marcus skips it. A run that
   never produced an experiment directory: `done configs/<name>.yaml --failed "<why>"`. Never
   remove a queue item any other way. Items that were queued but not started this hour stay
   `queued` for the next run.
J2. **TLDR** (the second decision): rewrite `data/tldr.md` as nested markdown bullets for someone
   who opens the site cold. Two levels only: a top-level bullet is one claim, its children are the
   numbers that support it. At most 5 words per bullet. No headings, no tables, no
   prose paragraphs. Order: who leads the benchmark, the closest alternative, what the latest runs
   added, what changed since the previous run. Numbers come from `data/experiments` exactly as in the
   data. Write it in ASD-STE100 (Simplified Technical English) at about 90% compliance: one idea
   per bullet, active voice, present tense, approved general words (`use` not `utilize`, `show`
   not `demonstrate`), no idioms.
J3. **Regenerate**: `uv run python -m jev_tracker.site_data` (reads `data/tldr.md`), then
   `cd site && npm ci && npm run build && cd ..`.
J4. **Check**: `uv run ruff check . && uv run ruff format --check . && uv run pytest -q -n auto`.
J5. **Push**: commit `data/` (the new experiments, `data/queue.json` with the finished items
   removed, `data/tldr.md`), any `crawler/` or `configs/` change since B4, `site/public/data` and
   `site/dist` (explicit paths, no `git add .`) with the title
   `Run <YYYY-MM-DD HH:MM UTC>: <n> candidates, <m> runs` and a body that has the triage
   counts per verdict, one line per run with kept-mass@50/200, $/run and its verification
   (`verified: <n> Modal calls finished, <m> rerankers x 75 cases` or the `failed` problems
   verbatim), the crawler failures if any, and `Closes #<n>` for each `evaluate` issue run.
   `git push origin HEAD:main`; if `main` moved meanwhile (a site click), `git pull --rebase
   origin main` once and push again. Never force-push.
J6. **Slack DM** to Marcus Dominguez-Kuhne (Slack user `U0BQQC4046P`), with the session's Slack
   tools, whenever a step pushed something or a step failed (a run that found nothing new sends
   no DM — the automation fires every hour). First line: `Pushed to main: <commit URL>` (or
   `Nothing pushed: <what failed>`). Then one line per model run (kept-mass@50/200, $/run,
   verified or failed with the problems), and the list of `proposed` items awaiting his approval
   on the site (label + note). No channel posts.
J7. **Nothing new** (nothing queued to run, and Track B found no candidates, issues or
   submissions): stop without a commit or a DM, and say so in the session's final message.

## What the automation never does

- Force-push, amend, rewrite history on `main`, open PRs, or edit `docs/SPEC.md`.
- Run more than 10 experiments or a config over the Kev-27B budget line.
- Run an experiment un-batched (see **Batching** above).
- Run a `needs_adapter` model Marcus has not approved on the site (or via
  `jev_tracker.evaluation_queue approve`).
- Print or commit a token.

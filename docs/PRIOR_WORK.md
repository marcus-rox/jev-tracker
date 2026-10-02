# Prior work for the crawler (R-8)

**2026-10-02 · survey for the R-8 crawler design.** Question: how do existing trackers find and
track new models / papers / mentions, and what should `python -m crawler` copy or avoid? Every
endpoint below was probed live from this Devin VM on 2026-10-02 (counts are from those calls); the
one claim not verified is marked *(unverified)*.

## 1. Sources

| Source | API endpoint | Auth | Rate limit | Date filter | Recommended query for Jev / Kev / Laya |
|---|---|---|---|---|---|
| Hugging Face Hub models | `GET https://huggingface.co/api/models?search=<term>&sort=createdAt&direction=-1&limit=100` (also `author=`, `filter=<pipeline tag>`) | none (token raises quota) | `ratelimit-policy` header: 500 req / 300 s anonymous, 1000 / 300 s with a token | none server-side; sort by `createdAt` (or `lastModified`) and stop at `since` | one call per term in `kev, jev, laya, systemone, system-one, noul, decision`; plus `filter=text-ranking&sort=createdAt`. `search` matches the **repo id only**, not the model card (`search=decision model` → 50 StartLux/Moral-Decision hits, nothing about Jev) |
| Hugging Face papers | `GET /api/papers/search?q=<term>`; `GET /api/daily_papers?date=YYYY-MM-DD` | none | same HF-wide policy | `daily_papers?date=`; `papers/search` has none | `papers/search?q=jev` (30 hits, all relevant: Jev-Mem, JEV-as-a-Judge, Chinese-Jev), `"system one"`, `"decision model"`. Fields: `paper.id` (= arXiv id), `githubRepo`, `upvotes` |
| Papers with Code | `paperswithcode.com/api/v1/…` | — | — | — | **dead**: 302 → `hf.co/papers/trending` since 2025-07 ([issue #116](https://github.com/paperswithcode/paperswithcode-data/issues/116)) |
| arXiv API | `GET https://export.arxiv.org/api/query?search_query=…&sortBy=submittedDate&sortOrder=descending&max_results=200` (Atom) | none | [ToU](https://info.arxiv.org/help/api/tou.html): 1 request / 3 s, one connection; ≤2000 results/page, 30k/query | `submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]` (GMT) in the query string | `(abs:jev OR abs:"system one" OR abs:"decision model" OR abs:laya) AND submittedDate:[<since> TO <now>]`. `abs:jev` alone had 56 papers in Sept 2026; `kev` is noisy (Kevin, Kevlar) — use `abs:"kev-4b" OR abs:jaredpalmer` |
| GitHub repo search | `GET /search/repositories?q=<terms>+in:name,description,readme+created:>YYYY-MM-DD&sort=updated` | optional; PAT raises limit | 10 req/min anonymous, 30 req/min authenticated; `X-RateLimit-*` headers | `created:>`, `pushed:>` qualifiers | `"jaredpalmer/kev" created:>since` (4 repos in Sept), `kev reranker`, `jev`, `laya`, `systemone`, `topic:reranker created:>since` (45 in Sept). |
| GitHub issues/PR search | `GET /search/issues?q=<term>+is:issue+created:>YYYY-MM-DD` | same | 30 req/min authenticated | `created:>`, `updated:>` | `"kev-4b" is:issue created:>since` (109 hits incl. `jaredpalmer/kev/issues/…`). Query must include `is:issue` or `is:pull-request` (422 otherwise) |
| GitHub code search | `GET /search/code?q=<term>` | **PAT required** (401 anonymous) | 10 req/min | **none** (no `created`/`pushed` qualifier) | `"jaredpalmer/kev"` → 3,208 files; unbounded and undated → see "Not worth it" |
| Hacker News (Algolia) | `GET https://hn.algolia.com/api/v1/search_by_date?query=<term>&tags=(story,comment)&numericFilters=created_at_i><epoch>` | none | not published on the API page; Algolia's FAQ says 10,000 req/h per IP *(unverified)* | `numericFilters=created_at_i>X` | quoted phrases only: `"kev-4b"` (1), `jaredpalmer` (71), `"liquid d1"` (23). Bare `jev` → 613k hits (fuzzy match), useless |
| Web — Tavily | `POST https://api.tavily.com/search` with header `X-Tavily-Access-Mode: keyless` ([docs](https://docs.tavily.com/documentation/keyless)) | **none** keyless (rate-limited, limit unpublished); free key = 1,000 credits/mo, then $0.008/credit | unpublished for keyless | `days`/`time_range` request fields | keyless `jaredpalmer kev reranker decision model` returned HF, X and YouTube hits on the first try from this VM |
| Web — Brave | `GET https://api.search.brave.com/res/v1/web/search?q=…&freshness=pd` | key + **credit card** on file | $5 free credit/month ≈ 1,000 requests at $5/1k ([pricing](https://api-dashboard.search.brave.com/documentation/pricing)) | `freshness=pd|pw|pm` | fallback only |
| Web — Serper | `POST https://google.serper.dev/search` | key | 2,500 free credits one-time, then $1.00→$0.30 / 1k, credits expire in 6 months | `tbs=qdr:d` | fallback only |
| Web — DuckDuckGo HTML/lite | `html.duckduckgo.com/html/?q=`, `lite.duckduckgo.com/lite/?q=` | none | bot wall: both returned **HTTP 202 "anomaly"** with 0 results on the first request from this VM | none | **avoid** (datacenter IPs are blocked; [ddgs #290](https://github.com/deedy5/duckduckgo_search/issues/290), [#304](https://github.com/deedy5/ddgs/issues/304)) |

## 2. Existing trackers

| Tracker | Polls | Dedupe | Cadence | Take-away |
|---|---|---|---|---|
| [karpathy/arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite) (`arxiv_daemon.py`) | arXiv API, `sortBy=lastUpdatedDate`, 2000 per run | sqlite dict keyed by arXiv id; re-store only if `_time` is newer; **stop after 3 consecutive pages with no new paper** | cron, daily | copy the early-stop and the id-keyed store; exit code 1 when nothing new so the caller skips downstream work |
| [antonkomarev/github-trending-archive](https://github.com/antonkomarev/github-trending-archive) | `github.com/trending` HTML per language | none — one JSON snapshot per day | GitHub Actions, hourly (retries page outages) | avoid: HTML scraping, 24-h trending window, not searchable by term |
| [Tenormusica2024/huggingface-daily-insights-api](https://github.com/Tenormusica2024/huggingface-daily-insights-api) | HF Hub models + arXiv + LMArena | daily snapshot; "new" = first seen in the snapshot diff, not Hub `createdAt` | GitHub Actions, 00:00 UTC | copy: `first_seen` is *our* first sighting, independent of the source's own timestamps |
| [do-me/trending-huggingface-models](https://github.com/do-me/trending-huggingface-models) | HF `?library=transformers.js&other=feature-extraction&sort=trending` | table re-export; ntfy daily/weekly/monthly | GitHub Actions, daily | one filtered Hub list + a push notification is enough for a niche class of models |

Convergence: everyone polls **one** source on a cron, keys on the source's native id, and stores a
snapshot or an id set. Gaps shared by all four (so they are design decisions, not inherited):
none searches by *term* across sources; none records *which query* matched (needed for triage);
none notices that a model id is unchanged while its **weights moved** (Kev-27B `main` got new
weights on 2026-09-30 while `jaredpalmer/kev-27b` stayed the same id — see `docs/PLAN.md`).

## 3. What to copy

- **Dedupe key per source**, written to `crawler/seen.jsonl` as `{key, first_seen}`:
  `hf:<repo_id>@<sha>` (a revision bump re-surfaces once), `arxiv:<id>` without version (shared
  by the arXiv and HF-papers sources), `gh:<full_name>` for repos, `gh:<html_url>` for issues,
  `hn:<objectID>`, `web:<url>` normalised (lower-case host, strip fragment, `utm_*`, trailing `/`).
- **Polling cadence**: once a day, `since = last_run − 2 days` (overlap is free because of the
  seen set); stop paging a source after one page with no new key (arxiv-sanity's early stop).
  Budget per run ≈ 8 terms × 5 sources ≈ 50 requests; arXiv at 1 req / 3 s is the slowest (~30 s).
- **Candidate schema** (`crawler/candidates/<date>.jsonl`, one object per line): the four fields
  R-8 names — `url`, `title`, `snippet`, `first_seen` — plus `source` (`hf|hf_papers|arxiv|github|hn|web`),
  `key` (the dedupe key above) and `query` (the term that matched). `source` and `query` are what
  the daily triage (R-9) sorts by; `key` makes `seen.jsonl` and `candidates/*.jsonl` joinable.
- **Terms**: `kev`, `jev`, `laya`, `systemone`, `"system one"`, `noul`, `"decision model"`,
  `reranker` (last two only on arXiv/HF-papers and GitHub `topic:`), quoted phrases on HN.
- **Web search**: Tavily keyless first (no key, works from the VM); if it rate-limits, the free
  Tavily key (1,000 credits/month ≫ ~10 queries/day). No Brave/Serper account needed.
- **Progress + exit code**: one `tqdm` bar over (source, term) pairs; exit 1 when no candidate was
  written so the automation can skip the PR.

## 4. Not worth it

- DuckDuckGo HTML/lite scraping — bot-walled from datacenter IPs on the first request.
- Papers with Code — shut down 2025-07; the domain redirects to HF papers/trending.
- GitHub code search for mentions — PAT-only, 10 req/min, no date qualifier, 3,208 undated
  files for `"jaredpalmer/kev"`; repo + issue search cover new activity with `created:>`.
- GitHub trending page — no API, HTML, per-language 24-h window, not searchable by term.
- arXiv category RSS (`rss.arxiv.org/atom/cs.IR`, 41 entries/day) — no query filter; the API's
  `submittedDate` does the same with a term.
- HF `search=` for multi-word concepts (`decision model`) — id-only matching, capped lists; use
  HF papers search and `filter=text-ranking` for the concept, `search=` only for names.
- Bare `jev` on HN (613k fuzzy hits) and bare `kev` on arXiv (Kevin, Kevlar).
- Paid search tiers, an HF token (500→1000 req / 300 s), or a database: ~50 requests/day and two
  JSONL files fit well inside every free limit.

## Internal context

Phase 1 (`docs/PLAN.md`) already established the two cases the crawler has to catch: a new Hub id
(Kev-4B, Laya, Liquid d1) and a weights change under an existing id (Kev-27B v2). The `hf:<id>@<sha>`
key covers both. Rox-research's `reranker_alts` had no crawler; candidates there came from Slack.

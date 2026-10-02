# Crawler prior work (R-8)

Short survey before building `crawler/`; the design decisions it fed are at the end.

| Tool | What it does | Fit for R-8 |
|---|---|---|
| Google Alerts / Talkwalker Alerts | Keyword alerts by email | Needs a mailbox to parse; no API; no code-host coverage |
| GitHub search API (`/search/repositories`, `pushed:>=`) | Repos matching text, filterable by push date | Exactly the "new since" query; 10 req/min unauthenticated, 30 with a token |
| Hugging Face Hub API (`/api/models?search=&sort=lastModified`) | Models by id substring, newest first | Covers Kev checkpoints and quantisations; `decision-model` is already a Hub tag |
| arXiv API (`export.arxiv.org/api/query`) | Atom feed, full-text terms, date-sorted | Covers papers; whitespace means OR, so terms must be ANDed explicitly |
| Papers with Code / Semantic Scholar APIs | Paper + code graph | Keyed on paper ids, weak on "a repo called kev"; Semantic Scholar needs a key for volume |
| DuckDuckGo HTML endpoint | Keyless web search | Best keyless web coverage; no dates; rate-limits bots with a 202 image challenge |
| SerpAPI / Brave / Tavily search APIs | Web search with JSON | Need a key and billing; deferred until DuckDuckGo proves insufficient |
| `gh`-based "awesome list" bots, `changedetection.io` | Watch known pages | Only re-visit known URLs; cannot discover new mentions |

## Decisions taken

- One module per source with one `search(query, since) -> list[Candidate]` seam; a source can be
  swapped (e.g. DuckDuckGo → Brave) without touching the run loop.
- Queries live in `crawler/queries.yaml`, one list per source, because the Hub only matches ids
  (phrases are useless there) while GitHub/arXiv/web take free text.
- Dedupe on `url` with an append-only `crawler/seen.jsonl`; `first_seen` is the crawler's clock,
  and the next run's default window starts at the last run's max `first_seen`.
- Web results carry no dates, so the seen set alone defines "new" for that source.
- A blocked source (DuckDuckGo 403/202 bot wall) returns `[]` and is logged; any other failure
  is reported per (source, query) and the run exits 1 at the end.

"""python -m crawler [--since YYYY-MM-DD] [--out crawler/candidates/] [--queries crawler/queries.yaml]

Runs every (source x query) from queries.yaml with tqdm (Slack only when SLACK_USER_TOKEN is set), adds the site's submissions from
requests/ (source `submitted`), dedupes against crawler/seen.jsonl,
appends the new keys to it and writes crawler/candidates/<YYYY-MM-DD>.jsonl. A failing
(source, query) is printed and skipped; the exit code is 1 at the end if any failed.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from tqdm import tqdm

from crawler import arxiv, github, hackernews, huggingface, slack, submitted, twitter, web
from crawler.contract import (
    SOURCES,
    Candidate,
    Source,
    append_seen,
    default_since,
    new_candidates,
    read_seen,
    utcnow,
)

HERE = Path(__file__).resolve().parent
SEARCHERS = {
    "github": github.search,
    "huggingface": huggingface.search,
    "arxiv": arxiv.search,
    "web": web.search,
    "twitter": twitter.search,
    "hackernews": hackernews.search,
    "slack": slack.search,
}


def load_queries(path: Path, skip: frozenset[Source] = frozenset()) -> list[tuple[Source, str]]:
    per_source: dict[str, list[str]] = yaml.safe_load(path.read_text())
    unknown = set(per_source) - set(SOURCES)
    if unknown:
        raise ValueError(f"{path}: unknown sources {sorted(unknown)}; expected {SOURCES}")
    return [
        (source, q) for source in SOURCES if source not in skip for q in per_source.get(source, [])
    ]


def crawl(jobs: list[tuple[Source, str]], since: datetime) -> tuple[list[Candidate], list[str]]:
    found: list[Candidate] = []
    failed: list[str] = []
    for source, query in tqdm(jobs, desc="crawl", unit="query"):
        try:
            found.extend(SEARCHERS[source](query, since))
        except Exception as e:  # boundary: one bad (source, query) must not stop the others
            failed.append(f"{source} {query!r}: {type(e).__name__}: {e}")
            tqdm.write(f"FAILED {failed[-1]}")
    return found, failed


def write_candidates(out_dir: Path, candidates: list[Candidate], now: datetime) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{now.date().isoformat()}.jsonl"
    with path.open("a") as f:
        for c in candidates:
            f.write(c.model_dump_json() + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m crawler", description=__doc__)
    parser.add_argument("--since", type=datetime.fromisoformat, help="YYYY-MM-DD (UTC)")
    parser.add_argument("--out", type=Path, default=HERE / "candidates")
    parser.add_argument("--queries", type=Path, default=HERE / "queries.yaml")
    parser.add_argument("--seen", type=Path, default=HERE / "seen.jsonl")
    parser.add_argument("--requests", type=Path, default=HERE.parent / "requests")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    now = utcnow()
    seen = read_seen(args.seen)
    since = args.since.replace(tzinfo=UTC) if args.since else default_since(seen, now)
    skip: frozenset[Source] = frozenset() if slack.configured() else frozenset({"slack"})
    if skip:
        print(f"{slack.TOKEN_ENV} is not set; skipping sources {sorted(skip)}")
    jobs = load_queries(args.queries, skip)
    print(f"since {since.isoformat()}  {len(jobs)} (source, query) jobs  {len(seen)} seen keys")

    found, failed = crawl(jobs, since)
    if args.requests.exists():
        found.extend(submitted.load(args.requests))
    fresh = new_candidates(found, seen)
    out = write_candidates(args.out, fresh, now)
    append_seen(args.seen, fresh)

    for source in SOURCES:
        n_found = sum(c.source == source for c in found)
        n_new = sum(c.source == source for c in fresh)
        print(f"{source:12s} found {n_found:4d}  new {n_new:4d}")
    print(f"wrote {len(fresh)} candidates to {out}; seen set now {len(seen) + len(fresh)}")
    if failed:
        print(f"{len(failed)} (source, query) jobs failed:\n  " + "\n  ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

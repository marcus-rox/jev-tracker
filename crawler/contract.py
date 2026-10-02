"""What every source returns and what the seen set remembers (R-8).

    source.search(query, since) -> list[Candidate]      one module per source
    Seen(url, key, first_seen)                          one line of crawler/seen.jsonl

Dedupe key is `key`: the url for every source except Hugging Face, where it is `hf:<id>@<sha>` so
new weights under an existing model id surface once more. `first_seen` is the crawler's clock
(UTC) when the key was first returned, never the source's own date; the default `--since` of the
next run is the max `first_seen`.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

Source = Literal["github", "huggingface", "arxiv", "web", "twitter", "hackernews", "submitted"]
SOURCES: tuple[Source, ...] = (
    "github",
    "huggingface",
    "arxiv",
    "web",
    "twitter",
    "hackernews",
    "submitted",
)
DEFAULT_WINDOW = timedelta(days=7)
HTTP_TIMEOUT_SECONDS = 30.0


class Candidate(BaseModel):
    model_config = {"frozen": True}

    source: Source
    url: str
    key: str
    title: str
    snippet: str
    first_seen: datetime
    query: str


class Seen(BaseModel):
    model_config = {"frozen": True}

    url: str
    key: str
    first_seen: datetime


def read_seen(path: Path) -> list[Seen]:
    if not path.exists():
        return []
    return [Seen.model_validate_json(line) for line in path.read_text().splitlines() if line]


def append_seen(path: Path, candidates: list[Candidate]) -> None:
    with path.open("a") as f:
        for c in candidates:
            f.write(Seen(url=c.url, key=c.key, first_seen=c.first_seen).model_dump_json() + "\n")


def default_since(seen: list[Seen], now: datetime) -> datetime:
    """Last run's max first_seen, else `now - DEFAULT_WINDOW`."""
    if seen:
        return max(s.first_seen for s in seen)
    return now - DEFAULT_WINDOW


def new_candidates(candidates: list[Candidate], seen: list[Seen]) -> list[Candidate]:
    """Drop keys already in the seen set and repeats within this run (first occurrence wins)."""
    known = {s.key for s in seen}
    fresh: list[Candidate] = []
    for c in candidates:
        if c.key in known:
            continue
        known.add(c.key)
        fresh.append(c)
    return fresh


def utcnow() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)

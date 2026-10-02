"""Hacker News stories and comments via the Algolia API, newest first, filtered at `since`.

Algolia fuzzy-matches bare words (`jev` hits 613k items), so queries.yaml quotes HN phrases.
A story's snippet is its link; a comment's is its text with HTML stripped (first 300 chars).
"""

from datetime import datetime
from html.parser import HTMLParser

import httpx

from crawler.contract import HTTP_TIMEOUT_SECONDS, Candidate, utcnow

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"
ITEM_URL = "https://news.ycombinator.com/item?id={id}"
TAGS = "(story,comment)"
HITS_PER_PAGE = 50
COMMENT_TAG = "comment"
SNIPPET_CHARS = 300


def search(query: str, since: datetime) -> list[Candidate]:
    params = {
        "query": query,
        "tags": TAGS,
        "numericFilters": f"created_at_i>={int(since.timestamp())}",
        "hitsPerPage": HITS_PER_PAGE,
    }
    response = httpx.get(SEARCH_URL, params=params, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return parse(response.json(), query, since, utcnow())


class _TextExtractor(HTMLParser):
    """Collects the text of an HTML fragment, one space per tag so `<p>` keeps words apart."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def strip_html(fragment: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(fragment)
    return " ".join("".join(extractor.parts).split())


def snippet(hit: dict) -> str:
    if COMMENT_TAG in hit["_tags"]:
        return strip_html(hit["comment_text"])[:SNIPPET_CHARS]
    return hit.get("url") or hit.get("story_url") or ""


def parse(payload: dict, query: str, since: datetime, first_seen: datetime) -> list[Candidate]:
    out: list[Candidate] = []
    for hit in payload["hits"]:
        if datetime.fromisoformat(hit["created_at"]) < since:
            continue
        url = ITEM_URL.format(id=hit["objectID"])
        out.append(
            Candidate(
                source="hackernews",
                url=url,
                key=url,
                title=hit.get("title") or hit.get("story_title") or "",
                snippet=snippet(hit),
                first_seen=first_seen,
                query=query,
            )
        )
    return out

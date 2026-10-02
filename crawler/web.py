"""DuckDuckGo HTML endpoint (no key, no date filter) parsed with the stdlib html.parser.

DuckDuckGo gives no dates, so `since` is unused and the seen set alone decides what is new. When
it answers 403, or 202 with its "bots use DuckDuckGo too" duck-picking challenge instead of
results, the source logs and returns [] rather than failing the run.
"""

import logging
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit

import httpx

from crawler.contract import HTTP_TIMEOUT_SECONDS, Candidate, utcnow

SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) jev-tracker-crawler"
BLOCKED_STATUSES = {403, 202}
REDIRECT_PARAM = "uddg"
TITLE_CLASS = "result__a"
SNIPPET_CLASS = "result__snippet"

log = logging.getLogger(__name__)


def search(query: str, since: datetime) -> list[Candidate]:
    response = httpx.get(
        SEARCH_URL,
        params={"q": query},
        headers={"User-Agent": USER_AGENT},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    if response.status_code in BLOCKED_STATUSES:
        log.warning(
            "web: DuckDuckGo returned %d (bot wall) for %r; returning no results",
            response.status_code,
            query,
        )
        return []
    response.raise_for_status()
    return parse(response.text, query, since, utcnow())


class _ResultParser(HTMLParser):
    """Collects (href, text) of result title and snippet anchors in document order."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._field: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr = dict(attrs)
        classes = (attr.get("class") or "").split()
        if TITLE_CLASS in classes:
            self.results.append({"url": _target_url(attr.get("href") or ""), "title": ""})
            self._field = "title"
        elif SNIPPET_CLASS in classes and self.results:
            self.results[-1]["snippet"] = ""
            self._field = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._field = None

    def handle_data(self, data: str) -> None:
        if self._field is not None:
            self.results[-1][self._field] = self.results[-1].get(self._field, "") + data


def _target_url(href: str) -> str:
    """DuckDuckGo links go through //duckduckgo.com/l/?uddg=<real url>; unwrap for dedupe."""
    redirect = parse_qs(urlsplit(href).query).get(REDIRECT_PARAM)
    return redirect[0] if redirect else href


def parse(html: str, query: str, since: datetime, first_seen: datetime) -> list[Candidate]:
    parser = _ResultParser()
    parser.feed(html)
    return [
        Candidate(
            source="web",
            url=r["url"],
            title=" ".join(r["title"].split()),
            snippet=" ".join(r.get("snippet", "").split()),
            first_seen=first_seen,
            query=query,
        )
        for r in parser.results
    ]

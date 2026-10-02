"""X / Twitter mentions via the same keyless Tavily call as `web.py`, restricted to `site:x.com`.

Tavily sometimes returns off-site pages for a site: query, so only x.com / twitter.com hosts are
kept. The 429 rate-limit case is handled in `web.tavily_search` (returns []).
"""

from datetime import datetime
from urllib.parse import urlsplit

from crawler.contract import Candidate, utcnow
from crawler.web import tavily_search

SITE_FILTER = " site:x.com"
HOSTS = {"x.com", "twitter.com", "www.x.com", "www.twitter.com"}


def search(query: str, since: datetime) -> list[Candidate]:
    now = utcnow()
    return parse({"results": tavily_search(query + SITE_FILTER, since, now)}, query, since, now)


def parse(payload: dict, query: str, since: datetime, first_seen: datetime) -> list[Candidate]:
    return [
        Candidate(
            source="twitter",
            url=result["url"],
            key=result["url"],
            title=" ".join(result["title"].split()),
            snippet=" ".join(result["content"].split()),
            first_seen=first_seen,
            query=query,
        )
        for result in payload["results"]
        if urlsplit(result["url"]).hostname in HOSTS
    ]

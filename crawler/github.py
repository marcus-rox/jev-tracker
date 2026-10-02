"""GitHub repository search. Unauthenticated unless GITHUB_TOKEN is set (10 vs 30 searches/min)."""

import os
from datetime import datetime

import httpx

from crawler.contract import HTTP_TIMEOUT_SECONDS, Candidate, utcnow

SEARCH_URL = "https://api.github.com/search/repositories"
PER_PAGE = 50
TOKEN_ENV = "GITHUB_TOKEN"


def search(query: str, since: datetime) -> list[Candidate]:
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get(TOKEN_ENV):
        headers["Authorization"] = f"Bearer {token}"
    params = {
        "q": f"{query} pushed:>={since.date().isoformat()}",
        "sort": "updated",
        "order": "desc",
        "per_page": PER_PAGE,
    }
    response = httpx.get(SEARCH_URL, params=params, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return parse(response.json(), query, since, utcnow())


def parse(payload: dict, query: str, since: datetime, first_seen: datetime) -> list[Candidate]:
    return [
        Candidate(
            source="github",
            url=item["html_url"],
            key=item["html_url"],
            title=item["full_name"],
            snippet=item.get("description") or "",
            first_seen=first_seen,
            query=query,
        )
        for item in payload["items"]
        if datetime.fromisoformat(item["pushed_at"]) >= since
    ]

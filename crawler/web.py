"""Tavily web search in keyless mode (no API key; rate-limited with HTTP 429).

Tavily filters by whole days back from now, so `since` becomes a `days` count. Results carry no
usable date, so the seen set alone decides what is new. A 429 (keyless rate limit) logs and
returns [] rather than failing the run; anything else raises. `twitter.py` reuses `tavily_search`.
"""

import logging
import math
from datetime import datetime

import httpx

from crawler.contract import HTTP_TIMEOUT_SECONDS, Candidate, utcnow

SEARCH_URL = "https://api.tavily.com/search"
HEADERS = {"Content-Type": "application/json", "X-Tavily-Access-Mode": "keyless"}
MAX_RESULTS = 10
TOPIC = "general"
MIN_DAYS = 1
SECONDS_PER_DAY = 86400
BLOCKED_STATUSES = {429}

log = logging.getLogger(__name__)


def search(query: str, since: datetime) -> list[Candidate]:
    now = utcnow()
    return parse({"results": tavily_search(query, since, now)}, query, since, now)


def tavily_search(query: str, since: datetime, now: datetime) -> list[dict]:
    """One keyless Tavily call; the raw `results` list, or [] on the 429 rate limit."""
    body = {
        "query": query,
        "max_results": MAX_RESULTS,
        "topic": TOPIC,
        "days": days_back(since, now),
    }
    response = httpx.post(SEARCH_URL, headers=HEADERS, json=body, timeout=HTTP_TIMEOUT_SECONDS)
    if response.status_code in BLOCKED_STATUSES:
        log.warning(
            "web: Tavily returned %d (keyless rate limit) for %r; returning no results",
            response.status_code,
            query,
        )
        return []
    response.raise_for_status()
    return response.json()["results"]


def days_back(since: datetime, now: datetime) -> int:
    return max(MIN_DAYS, math.ceil((now - since).total_seconds() / SECONDS_PER_DAY))


def parse(payload: dict, query: str, since: datetime, first_seen: datetime) -> list[Candidate]:
    return [
        Candidate(
            source="web",
            url=result["url"],
            key=result["url"],
            title=" ".join(result["title"].split()),
            snippet=" ".join(result["content"].split()),
            first_seen=first_seen,
            query=query,
        )
        for result in payload["results"]
    ]

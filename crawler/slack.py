"""Rox Slack messages via `search.messages`, newest first, filtered at `since`.

Needs a user token with `search:read` in SLACK_USER_TOKEN (search.messages does not accept bot
tokens); without it `__main__` skips the source. The key is the message permalink; the title is
`#channel · user`; the snippet is the first 300 characters of the text.
"""

import os
from datetime import UTC, datetime

import httpx

from crawler.contract import HTTP_TIMEOUT_SECONDS, Candidate, utcnow

SEARCH_URL = "https://slack.com/api/search.messages"
TOKEN_ENV = "SLACK_USER_TOKEN"
COUNT = 100
SNIPPET_CHARS = 300
DM_NAME = "dm"


def configured() -> bool:
    return bool(os.environ.get(TOKEN_ENV))


def search(query: str, since: datetime) -> list[Candidate]:
    response = httpx.get(
        SEARCH_URL,
        params={"query": f"{query} after:{since:%Y-%m-%d}", "count": COUNT, "sort": "timestamp"},
        headers={"Authorization": f"Bearer {os.environ[TOKEN_ENV]}"},
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("ok"):
        raise ValueError(f"search.messages {query!r}: {payload.get('error')}")
    return parse(payload, query, since, utcnow())


def parse(payload: dict, query: str, since: datetime, first_seen: datetime) -> list[Candidate]:
    out: list[Candidate] = []
    for match in payload["messages"]["matches"]:
        posted = datetime.fromtimestamp(float(match["ts"]), tz=UTC)
        if posted < since:
            continue
        channel = match["channel"].get("name") or DM_NAME
        out.append(
            Candidate(
                source="slack",
                url=match["permalink"],
                key=match["permalink"],
                title=f"#{channel} · {match.get('username') or match.get('user', '')}",
                snippet=" ".join(match["text"].split())[:SNIPPET_CHARS],
                first_seen=first_seen,
                query=query,
            )
        )
    return out

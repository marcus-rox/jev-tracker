"""Hugging Face Hub model search, newest modification first, cut at `since` client-side.

The dedupe key is `hf:<id>@<sha>` (sha = head commit of the model repo), so a weights update under
an existing id (Kev-27B v2) is reported once more while a metadata-only touch of a seen sha is not.
"""

from datetime import datetime

import httpx

from crawler.contract import HTTP_TIMEOUT_SECONDS, Candidate, utcnow

MODELS_URL = "https://huggingface.co/api/models"
MODEL_PAGE = "https://huggingface.co/{id}"
KEY = "hf:{id}@{sha}"
LIMIT = 100
EXPAND = ["sha", "lastModified", "tags"]


def search(query: str, since: datetime) -> list[Candidate]:
    params = {
        "search": query,
        "sort": "lastModified",
        "direction": -1,
        "limit": LIMIT,
        "expand[]": EXPAND,
    }
    response = httpx.get(MODELS_URL, params=params, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return parse(response.json(), query, since, utcnow())


def parse(
    payload: list[dict], query: str, since: datetime, first_seen: datetime
) -> list[Candidate]:
    return [
        Candidate(
            source="huggingface",
            url=MODEL_PAGE.format(id=model["id"]),
            key=KEY.format(id=model["id"], sha=model["sha"]),
            title=model["id"],
            snippet=", ".join(model.get("tags", [])),
            first_seen=first_seen,
            query=query,
        )
        for model in payload
        if datetime.fromisoformat(model["lastModified"]) >= since
    ]

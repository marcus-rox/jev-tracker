"""arXiv Atom API, all fields, newest update first, cut at `since` client-side.

arXiv ORs whitespace-separated terms, so `"decision model" reranker` becomes
`all:"decision model" AND all:reranker` (quoted phrases stay phrases).
"""

import shlex
import xml.etree.ElementTree as ET
from datetime import datetime

import httpx

from crawler.contract import HTTP_TIMEOUT_SECONDS, Candidate, utcnow

QUERY_URL = "https://export.arxiv.org/api/query"
MAX_RESULTS = 50
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def search(query: str, since: datetime) -> list[Candidate]:
    params = {
        "search_query": arxiv_query(query),
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending",
        "max_results": MAX_RESULTS,
    }
    response = httpx.get(QUERY_URL, params=params, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return parse(response.text, query, since, utcnow())


def arxiv_query(query: str) -> str:
    terms = shlex.split(query)
    return " AND ".join(f'all:"{t}"' if " " in t else f"all:{t}" for t in terms)


def parse(atom_xml: str, query: str, since: datetime, first_seen: datetime) -> list[Candidate]:
    feed = ET.fromstring(atom_xml)
    out: list[Candidate] = []
    for entry in feed.findall("a:entry", ATOM):
        updated = datetime.fromisoformat(entry.findtext("a:updated", "", ATOM))
        if updated < since:
            continue
        out.append(
            Candidate(
                source="arxiv",
                url=entry.findtext("a:id", "", ATOM).strip(),
                title=" ".join(entry.findtext("a:title", "", ATOM).split()),
                snippet=" ".join(entry.findtext("a:summary", "", ATOM).split()),
                first_seen=first_seen,
                query=query,
            )
        )
    return out

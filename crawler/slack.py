"""Rox Slack messages, from the Slack MCP search the daily session runs.

The crawler has no Slack credentials. The daily Devin session calls the Slack MCP tool
`slack_search_public_and_private` once per query in `queries.yaml` (`keywords=[query]`,
`filters="after:<since>"`, `sort="timestamp"`), saves each `results` text to
`<dir>/<query slug>.md`, and passes the directory as `python -m crawler --slack-results <dir>`.
This module parses those files. The key is the message permalink; the title is `#channel · user`;
the snippet is the first 300 characters of the text.
"""

import re
from datetime import UTC, datetime
from pathlib import Path

from crawler.contract import Candidate

SNIPPET_CHARS = 300
DM_NAME = "dm"
RESULT_SEP = "### Result "
FIELD = {
    "channel": re.compile(r"^Channel: (.+?) \(ID: ", re.M),
    "user": re.compile(r"^From: (.+?) <", re.M),
    "ts": re.compile(r"^Message_ts: (\d+\.\d+)", re.M),
    "permalink": re.compile(r"^Permalink: \[link\]\((\S+)\)", re.M),
}
TEXT = re.compile(r"^Text: ?\n(.*?)(?:\n---\s*$|\Z)", re.M | re.S)


def slug(query: str) -> str:
    """File stem for a query: `"system one"` -> `system-one`."""
    return re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")


def parse(text: str, query: str, since: datetime, first_seen: datetime) -> list[Candidate]:
    out: list[Candidate] = []
    for block in text.split(RESULT_SEP)[1:]:
        fields = {name: pattern.search(block) for name, pattern in FIELD.items()}
        body = TEXT.search(block)
        if any(m is None for m in fields.values()) or body is None:
            continue
        posted = datetime.fromtimestamp(float(fields["ts"].group(1)), tz=UTC)  # type: ignore[union-attr]
        if posted < since:
            continue
        channel = fields["channel"].group(1)  # type: ignore[union-attr]
        permalink = fields["permalink"].group(1).replace("\\/", "/")  # type: ignore[union-attr]
        out.append(
            Candidate(
                source="slack",
                url=permalink,
                key=permalink,
                title=f"{channel if channel.startswith('#') else '#' + DM_NAME} · "
                f"{fields['user'].group(1)}",  # type: ignore[union-attr]
                snippet=" ".join(body.group(1).split())[:SNIPPET_CHARS],
                first_seen=first_seen,
                query=query,
            )
        )
    return out


def load(
    results_dir: Path, queries: list[str], since: datetime, first_seen: datetime
) -> list[Candidate]:
    """Candidates from `<results_dir>/<slug(query)>.md` for each query; a missing file is skipped."""
    out: list[Candidate] = []
    for query in queries:
        path = results_dir / f"{slug(query)}.md"
        if path.exists():
            out.extend(parse(path.read_text(), query, since, first_seen))
    return out

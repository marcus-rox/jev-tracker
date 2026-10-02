"""Site submissions as candidates: requests/<date>/<id>.json (written by jev_tracker.server).

    load(requests_dir) -> list[Candidate]      source="submitted", key=url

Every file is returned every run; the seen set is what stops a link being triaged twice.
"""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from crawler.contract import Candidate

QUERY = "site form"
SNIPPET = "submitted from the site's Evaluate page"


class Submission(BaseModel):
    model_config = {"frozen": True}

    url: str
    submitted_at: datetime


def load(requests_dir: Path) -> list[Candidate]:
    candidates = []
    for path in sorted(requests_dir.rglob("*.json")):
        s = Submission.model_validate_json(path.read_text())
        candidates.append(
            Candidate(
                source="submitted",
                url=s.url,
                key=s.url,
                title=s.url,
                snippet=SNIPPET,
                first_seen=s.submitted_at,
                query=QUERY,
            )
        )
    return candidates

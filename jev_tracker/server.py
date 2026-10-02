"""R-7: serves site/dist and takes "evaluate this" submissions from the site's form.

    GITHUB_TOKEN=<PAT> uv run python -m jev_tracker.server [--port 8000] [--dist site/dist]

POST /api/requests {"url": "<one web link>"} commits requests/<YYYY-MM-DD>/<HHMMSS>_<slug>.json
to `main` through the GitHub Contents API; the daily run reads that folder (crawler/submitted.py).
"""

import argparse
import base64
import json
import os
import re
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from jev_tracker.experiment import REPO_DIR

CONTENTS_URL = "https://api.github.com/repos/marcus-rox/jev-tracker/contents"
BRANCH = "main"
REQUESTS_DIR = "requests"
TOKEN_ENV = "GITHUB_TOKEN"
API_PATH = "/api/requests"
DEFAULT_PORT = 8000
DEFAULT_DIST = REPO_DIR / "site" / "dist"
HTTP_TIMEOUT_SECONDS = 30.0
MAX_BODY_BYTES = 4096
MAX_SLUG_CHARS = 60
NOT_SLUG = re.compile(r"[^a-z0-9]+")


class Submission(BaseModel):
    model_config = {"frozen": True}

    url: str
    submitted_at: datetime


def parse_request(body: bytes) -> str:
    """The one web link in a POST body, or ValueError naming what was wrong with it."""
    try:
        payload = json.loads(body)
    except ValueError as e:
        raise ValueError(f"body is not JSON: {e}") from e
    url = payload.get("url") if isinstance(payload, dict) else None
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"expected {{'url': '<one web link>'}}, got {payload!r}")
    url = url.strip()
    if len(url.split()) != 1:
        raise ValueError(f"expected exactly one web link, got {url!r}")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"expected an http(s) link, got {url!r}")
    return url


def request_path(url: str, now: datetime) -> str:
    parts = urlsplit(url)
    slug = NOT_SLUG.sub("_", (parts.netloc + parts.path).lower()).strip("_")[:MAX_SLUG_CHARS]
    return f"{REQUESTS_DIR}/{now:%Y-%m-%d}/{now:%H%M%S}_{slug}.json"


def commit_request(url: str, now: datetime, token: str, client: httpx.Client) -> str:
    """Writes the submission file to `main`; returns its GitHub URL."""
    path = request_path(url, now)
    record = Submission(url=url, submitted_at=now).model_dump_json(indent=1) + "\n"
    response = client.put(
        f"{CONTENTS_URL}/{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={
            "message": f"request: evaluate {url}",
            "content": base64.b64encode(record.encode()).decode(),
            "branch": BRANCH,
        },
    )
    response.raise_for_status()
    return response.json()["content"]["html_url"]


class SiteHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, commit: Callable[[str, datetime], str], **kwargs) -> None:
        self.commit = commit
        super().__init__(*args, **kwargs)

    def do_POST(self) -> None:  # noqa: N802  (http.server's name)
        if self.path != API_PATH:
            self.send_error(HTTPStatus.NOT_FOUND, f"no POST route {self.path}")
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY_BYTES:
            self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body is {length} bytes")
            return
        try:
            url = parse_request(self.rfile.read(length))
        except ValueError as e:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            return
        try:
            html_url = self.commit(url, datetime.now(UTC).replace(microsecond=0))
        except httpx.HTTPError as e:  # boundary: report the GitHub failure to the browser
            self._json(HTTPStatus.BAD_GATEWAY, {"error": f"GitHub write failed: {e}"})
            return
        self._json(HTTPStatus.CREATED, {"url": url, "html_url": html_url})

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m jev_tracker.server", description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dist", type=Path, default=DEFAULT_DIST)
    args = parser.parse_args(argv)
    token = os.environ.get(TOKEN_ENV)
    if not token:
        print(f"{TOKEN_ENV} is not set; submissions need a GitHub token with Contents write")
        return 1
    if not (args.dist / "index.html").exists():
        print(f"{args.dist} has no index.html; run `cd site && npm run build` first")
        return 1
    client = httpx.Client(timeout=HTTP_TIMEOUT_SECONDS)
    commit = partial(commit_request, token=token, client=client)
    handler = partial(SiteHandler, directory=str(args.dist), commit=commit)
    print(f"serving {args.dist} on http://localhost:{args.port}  (POST {API_PATH})")
    ThreadingHTTPServer(("", args.port), handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""R-7: serves site/dist and takes "evaluate this" submissions from the site's form.

    GITHUB_TOKEN=<PAT> uv run python -m jev_tracker.server [--port 8000] [--dist site/dist]

POST /api/requests {"text": "<whatever was typed>"} commits requests/<YYYY-MM-DD>/<HHMMSS>_<slug>.json
to `main` through the GitHub Contents API; the next automation run reads that folder (crawler/submitted.py).
GET /api/requests returns every submission on `main` newest first (one GraphQL call), cached for
QUEUE_CACHE_SECONDS and cleared by each new submission.
GET /api/queue returns data/queue.json as it is on `main` right now (jev_tracker.evaluation_queue),
so the site shows the evaluation queue without a redeploy; cached for QUEUE_CACHE_SECONDS.
POST /api/queue/decide {"config": "configs/<name>.yaml", "decision": "approve" | "reject"} moves a
proposed item to queued (or drops it) and commits the new data/queue.json to `main`. A reject must
also carry {"phrase": "Skip this Run", "password": <QUEUE_SKIP_PASSWORD>}, else 400 / 403.
"""

import argparse
import base64
import hmac
import json
import os
import re
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from jev_tracker.evaluation_queue import Queue, approved, rejected
from jev_tracker.experiment import REPO_DIR

OWNER, REPO = "marcus-rox", "jev-tracker"
CONTENTS_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/contents"
GRAPHQL_URL = "https://api.github.com/graphql"
BRANCH = "main"
REQUESTS_DIR = "requests"
REQUEST_SUFFIX = ".json"
TOKEN_ENV = "GITHUB_TOKEN"
SKIP_PASSWORD_ENV = "QUEUE_SKIP_PASSWORD"
SKIP_PHRASE = "Skip this Run"
API_PATH = "/api/requests"
QUEUE_PATH = "/api/queue"
DECIDE_PATH = "/api/queue/decide"
QUEUE_FILE = "data/queue.json"
Decision = Literal["approve", "reject"]
DECISIONS: tuple[Decision, ...] = ("approve", "reject")
QUEUE_CACHE_SECONDS = 30.0
EMPTY_QUEUE = b'{"items": []}\n'
DEFAULT_PORT = 8000
DEFAULT_DIST = REPO_DIR / "site" / "dist"
HTTP_TIMEOUT_SECONDS = 30.0
MAX_BODY_BYTES = 4096
MAX_SLUG_CHARS = 60
NOT_SLUG = re.compile(r"[^a-z0-9]+")
NOT_SCHEME = re.compile(r"^https?://")


class Submission(BaseModel):
    model_config = {"frozen": True}

    text: str
    submitted_at: datetime


class Suggestion(BaseModel):
    """One submission as the site's Suggestions tab lists it; `path` is its file on `main`."""

    model_config = {"frozen": True}

    path: str
    text: str
    submitted_at: datetime


# requests/<date>/<file>.json is two levels deep, so the query nests Tree entries twice.
REQUESTS_QUERY = """
query($owner: String!, $name: String!, $expression: String!) {
  repository(owner: $owner, name: $name) {
    object(expression: $expression) {
      ... on Tree { entries { name object { ... on Tree { entries { name object { ... on Blob { text } } } } } } }
    }
  }
}
"""


def parse_request(body: bytes) -> str:
    """The submitted text in a POST body, or ValueError naming what was wrong with it."""
    try:
        payload = json.loads(body)
    except ValueError as e:
        raise ValueError(f"body is not JSON: {e}") from e
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"expected {{'text': '<non-empty text>'}}, got {payload!r}")
    return text.strip()


def parse_decision(body: bytes) -> tuple[Path, Decision]:
    """(config, decision) from a POST body, or ValueError naming what was wrong with it."""
    try:
        payload = json.loads(body)
    except ValueError as e:
        raise ValueError(f"body is not JSON: {e}") from e
    config = payload.get("config") if isinstance(payload, dict) else None
    decision = payload.get("decision") if isinstance(payload, dict) else None
    if not isinstance(config, str) or not config or decision not in DECISIONS:
        raise ValueError(
            f"expected {{'config': 'configs/<name>.yaml', 'decision': {'|'.join(DECISIONS)}}}, got {payload!r}"
        )
    return Path(config), decision


def check_skip(body: bytes, password: str | None) -> None:
    """ValueError unless the body types SKIP_PHRASE exactly; PermissionError unless its password matches."""
    payload = json.loads(body)
    if payload.get("phrase") != SKIP_PHRASE:
        raise ValueError(f"type {SKIP_PHRASE!r} to skip")
    if not password:
        raise PermissionError(f"{SKIP_PASSWORD_ENV} is not set on the server; Skip is disabled")
    typed = payload.get("password")
    if not isinstance(typed, str) or not hmac.compare_digest(typed.encode(), password.encode()):
        raise PermissionError("wrong password")


def decided(queue: Queue, config: Path, decision: Decision, now: datetime) -> Queue:
    return approved(queue, [config], now) if decision == "approve" else rejected(queue, [config])


def request_path(text: str, now: datetime) -> str:
    slug = NOT_SLUG.sub("_", NOT_SCHEME.sub("", text.lower())).strip("_")[:MAX_SLUG_CHARS]
    return f"{REQUESTS_DIR}/{now:%Y-%m-%d}/{now:%H%M%S}_{slug}.json"


def commit_request(text: str, now: datetime, token: str, client: httpx.Client) -> str:
    """Writes the submission file to `main`; returns its GitHub URL."""
    path = request_path(text, now)
    record = Submission(text=text, submitted_at=now).model_dump_json(indent=1) + "\n"
    response = client.put(
        f"{CONTENTS_URL}/{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json={
            "message": f"request: evaluate {text}",
            "content": base64.b64encode(record.encode()).decode(),
            "branch": BRANCH,
        },
    )
    response.raise_for_status()
    return response.json()["content"]["html_url"]


def suggestions(payload: dict) -> list[Suggestion]:
    """Every submission in a REQUESTS_QUERY response, newest first; [] when requests/ is absent."""
    tree = payload["data"]["repository"]["object"] or {"entries": []}
    found = []
    for day in tree["entries"]:
        for file in (day["object"] or {}).get("entries", []):
            if file["name"].endswith(REQUEST_SUFFIX):
                record = Submission.model_validate_json(file["object"]["text"])
                path = f"{REQUESTS_DIR}/{day['name']}/{file['name']}"
                found.append(Suggestion(path=path, **record.model_dump()))
    return sorted(found, key=lambda s: (s.submitted_at, s.path), reverse=True)


def fetch_suggestions(token: str, client: httpx.Client) -> bytes:
    """GET /api/requests's body: {"items": [Suggestion, ...]} read from `main` in one GraphQL call."""
    response = client.post(
        GRAPHQL_URL,
        headers={"Authorization": f"Bearer {token}"},
        json={
            "query": REQUESTS_QUERY,
            "variables": {"owner": OWNER, "name": REPO, "expression": f"{BRANCH}:{REQUESTS_DIR}"},
        },
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise httpx.HTTPError(f"GraphQL errors: {payload['errors']}")
    items = [s.model_dump(mode="json") for s in suggestions(payload)]
    return json.dumps({"items": items}).encode()


def fetch_queue(token: str, client: httpx.Client) -> bytes:
    """data/queue.json as committed on `main`; an empty queue when the file does not exist yet."""
    response = client.get(
        f"{CONTENTS_URL}/{QUEUE_FILE}",
        params={"ref": BRANCH},
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.raw+json"},
    )
    if response.status_code == HTTPStatus.NOT_FOUND:
        return EMPTY_QUEUE
    response.raise_for_status()
    return response.content


def fetch_queue_with_sha(token: str, client: httpx.Client) -> tuple[Queue, str | None]:
    """(queue, blob sha) of data/queue.json on `main`; sha None when the file does not exist yet."""
    response = client.get(
        f"{CONTENTS_URL}/{QUEUE_FILE}",
        params={"ref": BRANCH},
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    if response.status_code == HTTPStatus.NOT_FOUND:
        return Queue(), None
    response.raise_for_status()
    payload = response.json()
    return Queue.model_validate_json(base64.b64decode(payload["content"])), payload["sha"]


def commit_decision(
    config: Path, decision: Decision, now: datetime, token: str, client: httpx.Client
) -> Queue:
    """Applies the decision to data/queue.json on `main` and returns the new queue."""
    queue, sha = fetch_queue_with_sha(token, client)
    new_queue = decided(queue, config, decision, now)
    body = {
        "message": f"queue: {decision} {config}",
        "content": base64.b64encode((new_queue.model_dump_json(indent=1) + "\n").encode()).decode(),
        "branch": BRANCH,
    }
    if sha is not None:
        body["sha"] = sha
    response = client.put(
        f"{CONTENTS_URL}/{QUEUE_FILE}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
        json=body,
    )
    response.raise_for_status()
    return new_queue


class GitHubCache:
    """Serves one fetch per QUEUE_CACHE_SECONDS so page loads do not each hit GitHub."""

    def __init__(self, fetch: Callable[[], bytes], ttl: float = QUEUE_CACHE_SECONDS) -> None:
        self.fetch = fetch
        self.ttl = ttl
        self.body: bytes | None = None
        self.fetched_at = 0.0

    def get(self) -> bytes:
        if self.body is None or time.monotonic() - self.fetched_at > self.ttl:
            self.body = self.fetch()
            self.fetched_at = time.monotonic()
        return self.body

    def replace(self, body: bytes) -> None:
        self.body = body
        self.fetched_at = time.monotonic()

    def clear(self) -> None:
        self.body = None


class SiteHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args,
        commit: Callable[[str, datetime], str],
        decide: Callable[[Path, Decision, datetime], Queue],
        queue: GitHubCache,
        suggestions: GitHubCache,
        skip_password: str | None,
        **kwargs,
    ) -> None:
        self.commit = commit
        self.decide = decide
        self.queue = queue
        self.suggestions = suggestions
        self.skip_password = skip_password
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802  (http.server's name)
        cache = {QUEUE_PATH: self.queue, API_PATH: self.suggestions}.get(self.path)
        if cache is None:
            self._static = True
            super().do_GET()
            return
        try:
            body = cache.get()
        except httpx.HTTPError as e:  # boundary: report the GitHub failure to the browser
            self._json(HTTPStatus.BAD_GATEWAY, {"error": f"GitHub read failed: {e}"})
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", f"max-age={int(QUEUE_CACHE_SECONDS)}")
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802  (http.server's name)
        self._static = True
        super().do_HEAD()

    def end_headers(self) -> None:
        if getattr(self, "_static", False):
            path = urlsplit(self.path).path
            cache_control = (
                "public, max-age=31536000, immutable" if path.startswith("/assets/") else "no-cache"
            )
            self.send_header("Cache-Control", cache_control)
        super().end_headers()

    def do_POST(self) -> None:  # noqa: N802  (http.server's name)
        if self.path not in (API_PATH, DECIDE_PATH):
            self.send_error(HTTPStatus.NOT_FOUND, f"no POST route {self.path}")
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_BODY_BYTES:
            self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body is {length} bytes")
            return
        body = self.rfile.read(length)
        now = datetime.now(UTC).replace(microsecond=0)
        try:
            if self.path == API_PATH:
                text = parse_request(body)
                html_url = self.commit(text, now)
                self.suggestions.clear()
                self._json(HTTPStatus.CREATED, {"text": text, "html_url": html_url})
            else:
                config, decision = parse_decision(body)
                if decision == "reject":
                    check_skip(body, self.skip_password)
                queue = self.decide(config, decision, now)
                self.queue.replace(queue.model_dump_json(indent=1).encode() + b"\n")
                self._json(HTTPStatus.OK, queue.model_dump(mode="json"))
        except ValueError as e:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
        except PermissionError as e:
            self._json(HTTPStatus.FORBIDDEN, {"error": str(e)})
        except httpx.HTTPError as e:  # boundary: report the GitHub failure to the browser
            self._json(HTTPStatus.BAD_GATEWAY, {"error": f"GitHub write failed: {e}"})

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
    decide = partial(commit_decision, token=token, client=client)
    queue = GitHubCache(partial(fetch_queue, token=token, client=client))
    suggestions = GitHubCache(partial(fetch_suggestions, token=token, client=client))
    skip_password = os.environ.get(SKIP_PASSWORD_ENV)
    if not skip_password:
        print(f"{SKIP_PASSWORD_ENV} is not set; Skip on the site will be refused")
    handler = partial(
        SiteHandler,
        directory=str(args.dist),
        commit=commit,
        decide=decide,
        queue=queue,
        suggestions=suggestions,
        skip_password=skip_password,
    )
    print(
        f"serving {args.dist} on http://localhost:{args.port}  "
        f"(GET+POST {API_PATH}, GET {QUEUE_PATH}, POST {DECIDE_PATH})"
    )
    ThreadingHTTPServer(("", args.port), handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())

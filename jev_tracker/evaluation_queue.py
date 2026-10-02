"""The model evaluation queue: data/queue.json on `main`, shown live by the site.

    uv run python -m jev_tracker.evaluation_queue add crawler/triage/<date>.yaml   runnable -> queued, needs_adapter with a config -> proposed
    uv run python -m jev_tracker.evaluation_queue approve configs/<name>.yaml ...   proposed -> queued (the site's Approve button does the same)
    uv run python -m jev_tracker.evaluation_queue reject configs/<name>.yaml ...    drop proposed items (the site's Skip button)
    uv run python -m jev_tracker.evaluation_queue start configs/<name>.yaml ...     queued -> running
    uv run python -m jev_tracker.evaluation_queue done configs/<name>.yaml --experiment <id>   verified -> gone; otherwise -> failed with the evidence
    uv run python -m jev_tracker.evaluation_queue done configs/<name>.yaml --failed "<why>"   the run never produced an experiment
    uv run python -m jev_tracker.evaluation_queue show

One item per config. `proposed` is a model Devin wants Marcus's approval to run (usually one that
needs adapter work first; `note` says what); `queued` runs in the next daily run's budget; several
items may be running at once; a finished one leaves the queue and exists only as its rows in the
site data. `done` first runs `jev_tracker.experiment verify` on the experiment: an item leaves the
queue only when every Modal call finished, every reranker answered every case and the kept-mass
table exists; otherwise it becomes `failed` and carries that evidence until Marcus skips it. `data/queue.json` is the single source of truth: the daily run commits the `start`
state straight to `main` so the site shows it while runs are going, the site's server commits
approvals to `main` (jev_tracker.server), and the run's PR carries the `done` removal.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel

from crawler.contract import utcnow
from crawler.triage import TriageDecision, read_triage
from jev_tracker.experiment import REPO_DIR, Evidence, verify

QUEUE_PATH = REPO_DIR / "data" / "queue.json"
Status = Literal["proposed", "queued", "running", "failed"]
PROPOSED_BY: dict[str, Status] = {"runnable": "queued", "needs_adapter": "proposed"}


class QueueItem(BaseModel):
    model_config = {"frozen": True}

    config: Path
    label: str
    source: str
    url: str
    status: Status = "queued"
    note: str = ""
    queued_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    evidence: Evidence | None = None  # set on a failed item: what `verify` found


class Queue(BaseModel):
    items: list[QueueItem] = []


def read_queue(path: Path) -> Queue:
    return Queue.model_validate_json(path.read_text()) if path.exists() else Queue()


def write_queue(path: Path, queue: Queue) -> None:
    path.write_text(queue.model_dump_json(indent=1) + "\n")


HOST_SOURCE = {
    "github.com": "github",
    "huggingface.co": "huggingface",
    "arxiv.org": "arxiv",
    "x.com": "twitter",
    "twitter.com": "twitter",
    "news.ycombinator.com": "hackernews",
}


def source_of(key: str, url: str) -> str:
    """Crawler source a triage decision came from: `hf:` keys are Hugging Face, otherwise by host."""
    if key.startswith("hf:"):
        return "huggingface"
    host = urlsplit(url).hostname or ""
    if host.endswith(".slack.com"):
        return "slack"
    return HOST_SOURCE.get(host.removeprefix("www."), "web")


def enqueued(queue: Queue, decisions: list[TriageDecision], now: datetime) -> Queue:
    """`queue` plus one item per decision with a config not already in it: runnable -> queued,
    needs_adapter -> proposed (the config is the one Devin would write once approved)."""
    present = {item.config for item in queue.items}
    added: list[QueueItem] = []
    for d in decisions:
        if d.verdict not in PROPOSED_BY or d.config is None or d.config in present:
            continue
        present.add(d.config)
        added.append(
            QueueItem(
                config=d.config,
                label=d.config.stem,
                source=source_of(d.key, d.url),
                url=d.url,
                status=PROPOSED_BY[d.verdict],
                note=d.reason if d.verdict == "needs_adapter" else "",
                queued_at=now,
            )
        )
    return Queue(items=[*queue.items, *added])


def approved(queue: Queue, configs: list[Path], now: datetime) -> Queue:
    """`queue` with the given proposed items queued (approval time replaces the proposal time)."""
    return Queue(
        items=[
            item.model_copy(update={"status": "queued", "queued_at": now})
            if item.config in configs and item.status == "proposed"
            else item
            for item in queue.items
        ]
    )


DISMISSABLE: frozenset[Status] = frozenset({"proposed", "failed"})


def rejected(queue: Queue, configs: list[Path]) -> Queue:
    """Drop proposed items (not wanted) and failed ones (seen); the site's Skip button."""
    return Queue(
        items=[
            item
            for item in queue.items
            if not (item.config in configs and item.status in DISMISSABLE)
        ]
    )


def started(queue: Queue, configs: list[Path], now: datetime) -> Queue:
    """`queue` with the given configs marked running (added first if they were never queued)."""
    known = {item.config for item in queue.items}
    items = [
        item.model_copy(update={"status": "running", "started_at": now})
        if item.config in configs
        else item
        for item in queue.items
    ]
    for config in configs:
        if config not in known:
            items.append(
                QueueItem(
                    config=config,
                    label=config.stem,
                    source="manual",
                    url="",
                    status="running",
                    queued_at=now,
                    started_at=now,
                )
            )
    return Queue(items=items)


def finished(queue: Queue, configs: list[Path]) -> Queue:
    return Queue(items=[item for item in queue.items if item.config not in configs])


def failed(queue: Queue, config: Path, why: str, found: Evidence | None, now: datetime) -> Queue:
    """`queue` with `config` marked failed, carrying why and whatever `verify` found."""
    update = {"status": "failed", "note": why, "evidence": found, "finished_at": now}
    return Queue(
        items=[
            item.model_copy(update=update) if item.config == config else item
            for item in queue.items
        ]
    )


def done(
    queue: Queue, config: Path, found: Evidence | None, why: str | None, now: datetime
) -> Queue:
    """Verified experiment -> item gone; anything else -> failed with the evidence."""
    if found is None:
        return failed(queue, config, why or "the run produced no experiment", None, now)
    if found.ok:
        return finished(queue, [config])
    return failed(queue, config, "; ".join(found.problems), found, now)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m jev_tracker.evaluation_queue", description=__doc__
    )
    parser.add_argument("--queue", type=Path, default=QUEUE_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("add").add_argument("triage", type=Path)
    sub.add_parser("approve").add_argument("configs", type=Path, nargs="+")
    sub.add_parser("reject").add_argument("configs", type=Path, nargs="+")
    sub.add_parser("start").add_argument("configs", type=Path, nargs="+")
    done_cmd = sub.add_parser("done")
    done_cmd.add_argument("config", type=Path)
    done_cmd.add_argument("--experiment", help="data/experiments/<id> the run wrote")
    done_cmd.add_argument("--failed", dest="why", help="why there is no experiment to verify")
    sub.add_parser("show")
    args = parser.parse_args(argv)

    queue = read_queue(args.queue)
    now = utcnow()
    if args.command == "add":
        queue = enqueued(queue, read_triage(args.triage), now)
    elif args.command == "approve":
        queue = approved(queue, args.configs, now)
    elif args.command == "reject":
        queue = rejected(queue, args.configs)
    elif args.command == "start":
        queue = started(queue, args.configs, now)
    elif args.command == "done":
        found = None if args.experiment is None else verify(args.experiment)
        queue = done(queue, args.config, found, args.why, now)
    if args.command != "show":
        write_queue(args.queue, queue)
    for item in queue.items:
        print(
            f"{item.status:8} {item.label:40} {item.source:12} since {item.started_at or item.queued_at:%Y-%m-%d %H:%M}Z"
            + (f"  {item.note}" if item.note else "")
        )
    counts = ", ".join(f"{sum(i.status == s for i in queue.items)} {s}" for s in Status.__args__)
    print(f"{len(queue.items)} in queue ({counts})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

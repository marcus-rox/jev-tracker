"""The model evaluation queue: data/queue.json on `main`, shown live by the site.

    uv run python -m jev_tracker.evaluation_queue add crawler/triage/<date>.yaml   runnable decisions -> queued
    uv run python -m jev_tracker.evaluation_queue start configs/<name>.yaml ...     queued -> running
    uv run python -m jev_tracker.evaluation_queue done configs/<name>.yaml ...      drop finished items
    uv run python -m jev_tracker.evaluation_queue show

One item per config. Several items may be running at once; a finished one leaves the queue and
exists only as its rows in the site data. `data/queue.json` is the single source of truth: the
daily run commits the `start` state straight to `main` so the site shows it while runs are going,
and the run's PR carries the `done` removal.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from crawler.contract import utcnow
from crawler.triage import TriageDecision, read_triage
from jev_tracker.experiment import REPO_DIR

QUEUE_PATH = REPO_DIR / "data" / "queue.json"
Status = Literal["queued", "running"]


class QueueItem(BaseModel):
    model_config = {"frozen": True}

    config: Path
    label: str
    source: str
    url: str
    status: Status = "queued"
    queued_at: datetime
    started_at: datetime | None = None


class Queue(BaseModel):
    items: list[QueueItem] = []


def read_queue(path: Path) -> Queue:
    return Queue.model_validate_json(path.read_text()) if path.exists() else Queue()


def write_queue(path: Path, queue: Queue) -> None:
    path.write_text(queue.model_dump_json(indent=1) + "\n")


def enqueued(queue: Queue, decisions: list[TriageDecision], now: datetime) -> Queue:
    """`queue` plus one queued item per runnable decision whose config is not already in it."""
    present = {item.config for item in queue.items}
    added = [
        QueueItem(
            config=d.config,
            label=d.config.stem,
            source=d.key.split(":")[0] if ":" in d.key else "submitted",
            url=d.url,
            queued_at=now,
        )
        for d in decisions
        if d.verdict == "runnable" and d.config is not None and d.config not in present
    ]
    return Queue(items=[*queue.items, *added])


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m jev_tracker.evaluation_queue", description=__doc__
    )
    parser.add_argument("--queue", type=Path, default=QUEUE_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("add").add_argument("triage", type=Path)
    sub.add_parser("start").add_argument("configs", type=Path, nargs="+")
    sub.add_parser("done").add_argument("configs", type=Path, nargs="+")
    sub.add_parser("show")
    args = parser.parse_args(argv)

    queue = read_queue(args.queue)
    now = utcnow()
    if args.command == "add":
        queue = enqueued(queue, read_triage(args.triage), now)
    elif args.command == "start":
        queue = started(queue, args.configs, now)
    elif args.command == "done":
        queue = finished(queue, args.configs)
    if args.command != "show":
        write_queue(args.queue, queue)
    for item in queue.items:
        print(
            f"{item.status:8} {item.label:40} {item.source:12} since {item.started_at or item.queued_at:%Y-%m-%d %H:%M}Z"
        )
    print(
        f"{len(queue.items)} in queue ({sum(i.status == 'running' for i in queue.items)} running)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

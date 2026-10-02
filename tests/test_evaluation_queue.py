from datetime import UTC, datetime
from pathlib import Path

from crawler.triage import TriageDecision
from jev_tracker.evaluation_queue import (
    Queue,
    approved,
    enqueued,
    finished,
    read_queue,
    rejected,
    started,
    write_queue,
)

NOW = datetime(2026, 10, 2, 13, 0, tzinfo=UTC)
LATER = datetime(2026, 10, 2, 13, 5, tzinfo=UTC)
RUNNABLE = TriageDecision(
    key="hf:org/kev-9b@abc",
    url="https://huggingface.co/org/kev-9b",
    verdict="runnable",
    reason="Kev checkpoint",
    config=Path("configs/kev9b_v2.yaml"),
)
SKIPPED = TriageDecision(key="https://x", url="https://x", verdict="not_jev", reason="a person")
PROPOSED = TriageDecision(
    key="hf:acme/clef-9b@def",
    url="https://huggingface.co/acme/clef-9b",
    verdict="needs_adapter",
    reason="own serving stack; needs a clef source",
    config=Path("configs/clef9b.yaml"),
)
NO_CONFIG = TriageDecision(
    key="https://y", url="https://y", verdict="needs_adapter", reason="GGUF export"
)


def test_lifecycle_queued_running_gone(tmp_path: Path) -> None:
    queue = enqueued(Queue(), [RUNNABLE, SKIPPED], NOW)
    assert [(i.label, i.status, i.source) for i in queue.items] == [
        ("kev9b_v2", "queued", "huggingface")
    ]
    assert enqueued(queue, [RUNNABLE], LATER) == queue  # same config is not queued twice

    queue = started(queue, [RUNNABLE.config, Path("configs/manual.yaml")], LATER)
    assert [(i.label, i.status, i.started_at) for i in queue.items] == [
        ("kev9b_v2", "running", LATER),
        ("manual", "running", LATER),
    ]

    path = tmp_path / "queue.json"
    write_queue(path, queue)
    assert read_queue(path) == queue
    assert finished(queue, [RUNNABLE.config]).items == queue.items[1:]
    assert read_queue(tmp_path / "missing.json") == Queue()


def test_proposed_needs_approval_before_it_is_queued() -> None:
    queue = enqueued(Queue(), [PROPOSED, NO_CONFIG, RUNNABLE, PROPOSED], NOW)  # one item per config
    assert [(i.label, i.status, i.note) for i in queue.items] == [
        ("clef9b", "proposed", "own serving stack; needs a clef source"),
        ("kev9b_v2", "queued", ""),
    ]
    assert approved(queue, [RUNNABLE.config], LATER) == queue  # only proposed items change
    approved_queue = approved(queue, [PROPOSED.config], LATER)
    assert [(i.status, i.queued_at) for i in approved_queue.items] == [
        ("queued", LATER),
        ("queued", NOW),
    ]
    assert rejected(queue, [PROPOSED.config, RUNNABLE.config]).items == queue.items[1:]
    assert read_queue(Path("data/queue.json")).items is not None  # the committed file still parses

from datetime import UTC, datetime
from pathlib import Path

from crawler.triage import TriageDecision
from jev_tracker.evaluation_queue import Queue, enqueued, finished, read_queue, started, write_queue

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


def test_lifecycle_queued_running_gone(tmp_path: Path) -> None:
    queue = enqueued(Queue(), [RUNNABLE, SKIPPED], NOW)
    assert [(i.label, i.status, i.source) for i in queue.items] == [("kev9b_v2", "queued", "hf")]
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

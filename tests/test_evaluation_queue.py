from datetime import UTC, datetime
from pathlib import Path

from crawler.triage import TriageDecision
from jev_tracker.evaluation_queue import (
    Queue,
    approved,
    done,
    enqueued,
    finished,
    read_queue,
    rejected,
    started,
    write_queue,
)
from jev_tracker.experiment import Evidence, evidence

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


VERIFIED = (
    "2026_10_02_03_29_11_above-dog"  # committed: 2 scored rerankers x 75 cases, table written
)


def test_evidence_passes_a_complete_committed_experiment_and_names_every_gap() -> None:
    good = evidence(VERIFIED, {"fc-1": "finished", "fc-2": "finished"})
    assert good.ok and good.kept_mass
    assert [(r.name, r.cases) for r in good.rerankers] == [
        ("jev_noul", 75),
        ("jev_score", 75),
        ("kev08b_noul", 75),
        ("kev08b_score", 75),
    ]

    bad = evidence(VERIFIED, {"fc-1": "FAILED: OOM", "fc-2": "running"})
    assert bad.problems == ["modal call fc-1: FAILED: OOM", "modal call fc-2: running"]
    assert bad.rerankers == good.rerankers  # the answers on disk are judged either way


def test_done_keeps_an_unverified_run_as_failed_until_skipped(tmp_path: Path) -> None:
    queue = started(enqueued(Queue(), [RUNNABLE], NOW), [RUNNABLE.config], LATER)
    good = evidence(VERIFIED, {"fc-1": "finished"})
    assert done(queue, RUNNABLE.config, good, None, LATER).items == []  # verified -> gone

    crashed = done(queue, RUNNABLE.config, None, "modal deploy failed", LATER)
    assert [(i.status, i.note, i.evidence, i.finished_at) for i in crashed.items] == [
        ("failed", "modal deploy failed", None, LATER)
    ]
    found = Evidence(
        experiment="x",
        calls={"fc": "running"},
        rerankers=[],
        kept_mass=False,
        problems=["modal call fc: running"],
    )
    partial = done(queue, RUNNABLE.config, found, None, LATER)
    assert (partial.items[0].evidence, partial.items[0].note) == (found, "modal call fc: running")
    path = tmp_path / "queue.json"
    write_queue(path, partial)
    assert read_queue(path) == partial
    assert finished(partial, [RUNNABLE.config]).items == []
    assert rejected(partial, [RUNNABLE.config]).items == []  # Skip clears a failed item too

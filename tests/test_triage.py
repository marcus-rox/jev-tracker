"""R-9: crawler/triage/<date>.yaml holds one decision per candidate and `check` verifies it."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from crawler.contract import Candidate
from crawler.triage import TriageDecision, main, read_triage, write_triage

NOW = datetime(2026, 10, 2, 0, 40, tzinfo=UTC)
CANDIDATES = [
    Candidate(
        source="github",
        url="https://github.com/a/kev-fork",
        key="https://github.com/a/kev-fork",
        title="a/kev-fork",
        snippet="",
        first_seen=NOW,
        query="kev",
    ),
    Candidate(
        source="web",
        url="https://example.com/jev-post",
        key="https://example.com/jev-post",
        title="post",
        snippet="",
        first_seen=NOW,
        query="jev",
    ),
]
RUNNABLE = TriageDecision(
    key="https://github.com/a/kev-fork",
    url="https://github.com/a/kev-fork",
    verdict="runnable",
    reason="Kev checkpoint, kev source",
    config=Path("configs/kev_fork.yaml"),
)
NOT_JEV = TriageDecision(
    key="https://example.com/jev-post",
    url="https://example.com/jev-post",
    verdict="not_jev",
    reason="blog post, no model",
)


def write_candidates(path: Path, candidates: list[Candidate]) -> None:
    path.write_text("".join(c.model_dump_json() + "\n" for c in candidates))


def test_R9_triage_round_trip(tmp_path: Path):
    path = tmp_path / "triage" / "2026-10-02.yaml"
    write_triage(path, [RUNNABLE, NOT_JEV])
    assert read_triage(path) == [RUNNABLE, NOT_JEV]
    assert "config: configs/kev_fork.yaml" in path.read_text()
    assert "config: null" in path.read_text()
    path.write_text("key: x")
    with pytest.raises(ValueError, match="expected a YAML list"):
        read_triage(path)


def test_R9_check_lists_missing_decision_and_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    candidates_path = tmp_path / "2026-10-02.jsonl"
    triage_path = tmp_path / "2026-10-02.yaml"
    write_candidates(candidates_path, CANDIDATES)
    write_triage(triage_path, [RUNNABLE])
    assert main(["check", str(triage_path), str(candidates_path), "--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "no decision: https://example.com/jev-post" in out
    assert (
        "runnable without config: https://github.com/a/kev-fork (config: configs/kev_fork.yaml)"
        in out
    )


def test_R9_check_passes_when_complete(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    candidates_path = tmp_path / "2026-10-02.jsonl"
    triage_path = tmp_path / "2026-10-02.yaml"
    write_candidates(candidates_path, CANDIDATES)
    write_triage(triage_path, [RUNNABLE, NOT_JEV])
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "kev_fork.yaml").write_text("name: kev_fork\n")
    assert main(["check", str(triage_path), str(candidates_path), "--root", str(tmp_path)]) == 0
    assert (
        capsys.readouterr().out.strip()
        == "2 decisions for 2 candidates: runnable 1  needs_adapter 0  not_jev 1"
    )

"""answer_diff: per-child |a-b| stats over the intersection of two raw files."""

import gzip
import json
from pathlib import Path

import pytest

from jev_tracker.answer_diff import child_answers, diff


def _raw(path: Path, noul: dict[int, float]) -> None:
    """One record whose children i get answer noul[i] (question id iN)."""
    record = {
        "case_id": "case_0",
        "batch": 0,
        "model": "m",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "latency_s": 0.1,
        "answers": {f"i{i}": {"type": "noul", "noul": v} for i, v in noul.items()},
        "scores": [
            {"parent_index": 0, "child_index": i, "question": f"i{i}", "option": None} for i in noul
        ],
    }
    with gzip.open(path, "wt") as f:
        json.dump({"run": "m", "written_at": "t", "records": [record]}, f)


def test_child_answers_keys_on_case_parent_child_option(tmp_path: Path) -> None:
    p = tmp_path / "a.json.gz"
    _raw(p, {0: 0.1, 1: 0.9})
    assert child_answers(p) == {("case_0", 0, 0, None): 0.1, ("case_0", 0, 1, None): 0.9}


def test_diff_stats_over_the_intersection(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    _raw(a, {0: 0.0, 1: 0.5, 2: 0.9})
    _raw(b, {0: 0.1, 1: 0.4, 3: 1.0})
    stats = diff(a, b)
    assert stats["n"] == 2 and stats["n_a"] == 3 and stats["n_b"] == 3
    assert stats["max"] == pytest.approx(0.1)
    assert stats["mean"] == pytest.approx(0.1)
    assert stats["p99"] == pytest.approx(0.1)

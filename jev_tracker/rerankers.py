"""The evaluation-side view of a reranker: per-child scores for one case, higher = ranked higher.

Model execution never happens here. A model run (Jev over HTTP, modal_app.py in-process on a
GPU) produces raw System One answers (systemone.RawRecord), kept as an experiment's
raw_<reranker>_<id>.json.gz; `ScoredReranker` turns those into scores with its Method. Production's
frozen ranking is read straight from the dataset.
"""

import gzip
import time
from pathlib import Path
from typing import Protocol

from jev_tracker.contract import ChildRef, DataPoint
from jev_tracker.methods import Binding, Method
from jev_tracker.systemone import RawFile, RawRecord


def read_raw(path: Path) -> list[RawRecord]:
    """A raw_<reranker>_<id>.json answer file, gzipped (.json.gz, as committed) or plain."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return RawFile.model_validate_json(f.read()).records
    return RawFile.model_validate_json(path.read_text()).records


def write_raw(path: Path, run: str, records: list[RawRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = RawFile(run=run, written_at=at, records=records).model_dump_json()
    if path.suffix == ".gz":
        with gzip.open(path, "wt", compresslevel=9) as f:
            f.write(text)
    else:
        path.write_text(text)
    return path


def read_jsonl_gz(path: Path) -> list[RawRecord]:
    """A worker's append-only shard on the Modal Volume (resumable, one request per line)."""
    if not path.exists():
        return []
    with gzip.open(path, "rt") as f:
        return [RawRecord.model_validate_json(line) for line in f if line.strip()]


def append_jsonl_gz(path: Path, rec: RawRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "at") as f:
        f.write(rec.model_dump_json() + "\n")


class Reranker(Protocol):
    name: str

    def scores(self, case: DataPoint) -> dict[ChildRef, float]: ...


class ProductionReranker:
    """Production's frozen order: score = -rank, so the ranking is reproduced exactly."""

    def __init__(self, name: str) -> None:
        self.name = name

    def scores(self, case: DataPoint) -> dict[ChildRef, float]:
        return {ref: -float(rank) for rank, ref in enumerate(case.production_output.ranking)}


class ScoredReranker:
    """A model's cached raw answers, scored by its Method. Crashes if a case has a child unanswered."""

    def __init__(self, name: str, method: Method, path: Path) -> None:
        self.name = name
        self.method = method
        self._scores: dict[str, dict[ChildRef, float]] = {}
        for rec in read_raw(path):
            case = self._scores.setdefault(rec.case_id, {})
            for s in rec.scores:
                ref = ChildRef(parent_index=s.parent_index, child_index=s.child_index)
                if ref in case:
                    raise ValueError(f"{path}: duplicate answer for {rec.case_id} {ref}")
                case[ref] = method.score(
                    rec.answers[s.question], Binding(question=s.question, option=s.option)
                )

    def scores(self, case: DataPoint) -> dict[ChildRef, float]:
        got = self._scores.get(case.id, {})
        n = sum(len(p.children) for p in case.input.parents)
        if len(got) != n:
            raise ValueError(f"{self.name}: {case.id} has {len(got)} of {n} children scored")
        return got

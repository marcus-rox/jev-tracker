"""Fan-out latency runs: worker assignment, dispatcher timing overwrite, sweep stats, validation."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from jev_tracker.contract import load_cases
from jev_tracker.experiment import Experiment, apply_dispatch, sweep_stats
from jev_tracker.methods import batches
from jev_tracker.modal_app import (
    _dispatch_case,
    _fanout_pass_item,
    fanout_plan,
)
from jev_tracker.systemone import RawRecord, Usage


def test_fanout_assigns_request_j_to_worker_j_mod_shards() -> None:
    assert fanout_plan(7, 3) == [[0, 3, 6], [1, 4], [2, 5]]
    assert fanout_plan(2, 4) == [[0], [1], [], []]


def _record(case_id: str, batch: int, started_at_s: float = 100.0) -> RawRecord:
    return RawRecord(
        case_id=case_id,
        batch=batch,
        model="kev-0.5b",
        usage=Usage(input_tokens=1, output_tokens=1),
        latency_s=9.9,
        answers={},
        scores=[],
        started_at_s=started_at_s,
    )


def _dispatched(rows: list[dict], timed_from: float = 50.0) -> dict:
    return {"timed_from": timed_from, "rows": rows}


def test_apply_dispatch_drops_warmup_and_shares_split_times() -> None:
    records = [
        _record("case_1", 0, started_at_s=10.0),  # warm-up duplicate: before timed_from
        _record("case_1", 0),
        _record("case_1", 0),  # a 422-split pair: same key, shares the parent's times
        _record("case_1", 1),
    ]
    dispatched = _dispatched(
        [
            {"case_id": "case_1", "batch": 0, "sent": 10.0, "recv": 10.5, "warmup": True},
            {"case_id": "case_1", "batch": 0, "sent": 100.0, "recv": 101.5, "warmup": False},
            {"case_id": "case_1", "batch": 1, "sent": 100.0, "recv": 103.0, "warmup": False},
        ]
    )
    out = apply_dispatch(records, dispatched)
    assert [(r.started_at_s, r.latency_s) for r in out] == [
        (100.0, 1.5),
        (100.0, 1.5),
        (100.0, 3.0),
    ]


def test_apply_dispatch_rejects_a_record_the_dispatcher_never_sent() -> None:
    with pytest.raises(ValueError, match="no dispatch record"):
        apply_dispatch([_record("case_1", 0)], _dispatched([]))


def test_apply_dispatch_rejects_a_timed_row_with_no_raw_record() -> None:
    dispatched = _dispatched(
        [
            {"case_id": "case_1", "batch": 0, "sent": 100.0, "recv": 101.0, "warmup": False},
            {"case_id": "case_1", "batch": 1, "sent": 100.0, "recv": 101.0, "warmup": False},
        ]
    )
    with pytest.raises(ValueError, match="no raw record"):
        apply_dispatch([_record("case_1", 0)], dispatched)


def _rep(
    wall_s: float,
    warmup: bool = False,
    requests: int = 1,
    children: int = 1,
    children_sent: int | None = None,
) -> dict:
    return {
        "requests": requests,
        "children": children,
        "rep": 0,
        "case_id": "case_1",
        "children_sent": children * requests if children_sent is None else children_sent,
        "records": requests,
        "wall_s": wall_s,
        "warmup": warmup,
    }


def test_sweep_stats_averages_timed_reps_and_ratios_the_1x1_point() -> None:
    rows = [
        _rep(99.0, warmup=True),  # ignored
        _rep(1.0),
        _rep(3.0),
        _rep(2.0, requests=2, children=8),
        _rep(4.0, requests=2, children=8),
    ]
    stats = {pt["children"] * pt["requests"]: pt for pt in sweep_stats(rows)}
    one = stats[1]
    assert one["n"] == 2 and one["p50_s"] == 3.0 and one["p50_ratio"] == 1.0
    assert not one["short"] and one["children_sent_min"] == one["children_sent_max"] == 1
    big = stats[16]
    assert big["n"] == 2 and big["p50_s"] == 4.0 and big["p50_ratio"] == pytest.approx(4 / 3)
    assert big["children_per_s"] == pytest.approx(16 / 4.0)


def test_sweep_stats_flags_a_point_whose_batches_were_cut_short() -> None:
    rows = [_rep(1.0), _rep(2.0, requests=2, children=8, children_sent=9)]
    stats = {pt["requests"]: pt for pt in sweep_stats(rows)}
    assert stats[2]["short"] and stats[2]["children_sent_min"] == stats[2]["children_sent_max"] == 9


def test_sweep_stats_counts_reps_where_a_request_split_into_more_records() -> None:
    rows = [_rep(1.0), {**_rep(2.0, requests=4, children=12), "records": 7}]
    stats = {pt["requests"]: pt for pt in sweep_stats(rows)}
    assert stats[1]["split_reps"] == 0 and stats[4]["split_reps"] == 1


def test_laya_fanout_validates_and_bekko_is_rejected() -> None:
    cfg = {
        "name": "x",
        "rerankers": {
            "prod": {"source": "production"},
            "laya": {"source": "laya", "method": "noul_query_in_state", "fanout": True},
        },
    }
    Experiment.model_validate(cfg)
    cfg["rerankers"]["bekko"] = {
        "source": "bekko",
        "method": "noul_query_in_state",
        "model": "hotchpotch/bekko-system-one-v0-17m",
        "fanout": True,
    }
    with pytest.raises(ValidationError, match="bekko cannot fan out"):
        Experiment.model_validate(cfg)


class _FakeQueue:
    """Enough of modal.Queue for `_dispatch_case`: puts land per partition, and canned
    acks / done tuples answer the reads."""

    def __init__(self, n_workers: int, dones: list[tuple]) -> None:
        self.puts: dict[str, list] = {}
        self._acks = [("reset_ok", w) for w in range(n_workers)]
        self._dones = list(dones)

    def put(self, v, partition=None) -> None:
        self.puts.setdefault(partition, []).append(v)

    def put_many(self, vs, partition=None) -> None:
        self.puts.setdefault(partition, []).extend(vs)

    def get_many(self, n, partition=None, block=True, timeout=None) -> list:
        if partition == "ack":
            return self._acks
        if partition == "done":
            dones, self._dones = self._dones, []
            return dones
        raise AssertionError(partition)


def _exp_run(src: dict):
    exp = Experiment.model_validate(
        {"name": "x", "rerankers": {"prod": {"source": "production"}, "m": src}}
    )
    return exp.scoring_run("e", "m", exp.rerankers["m"])


def test_dispatch_sends_one_item_per_worker_carrying_all_its_batches() -> None:
    case = load_cases()[0]
    run = _exp_run(
        {
            "source": "kev",
            "method": "noul_query_in_state",
            "shards": 4,
            "fanout": True,
        }
    )
    m = run.max_items
    n_requests = len(batches(case.input, run.max_items, run.max_chars))
    dones = [(case.id, bi, 0.0, 0.1, 1) for bi in range(n_requests)]
    q = _FakeQueue(run.shards, dones)
    with ThreadPoolExecutor(4) as pool:
        rows = _dispatch_case(run, q, case, False, pool)
    worker_items = [v for w in range(run.shards) for v in q.puts.get(f"w{w}", []) if len(v) == 3]
    assert all(item[0] == case.id and isinstance(item[1], list) for item in worker_items)
    assert sorted(bi for item in worker_items for bi in item[1]) == list(range(n_requests))
    assert len(worker_items) == run.shards
    for w in range(run.shards):
        item = next(v for v in q.puts[f"w{w}"] if len(v) == 3)
        assert item[1] == [j for j in range(n_requests) if j % run.shards == w]
        assert item[2] == m
        sent = {rows[j]["sent"] for j in item[1]}
        assert len(sent) == 1
    assert {r["batch"] for r in rows} == set(range(n_requests))


def _echo_predict(bodies: list[dict]) -> list[dict]:
    return [
        {
            "model": "laya",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "answers": {k: {"type": "noul", "noul": 0.5} for k in body["questions"]},
        }
        for body in bodies
    ]


def test_pass_worker_chunks_by_forward_batch_and_emits_one_done_per_batch() -> None:
    case = load_cases()[0]
    run = _exp_run(
        {
            "source": "laya",
            "method": "noul_query_in_state",
            "shards": 4,
            "fanout": True,
            "forward_batch": 64,
        }
    )
    calls: list[list[dict]] = []

    def predict(bodies: list[dict]) -> list[dict]:
        calls.append(bodies)
        return _echo_predict(bodies)

    records, dones = _fanout_pass_item(run, case, list(range(130)), 1, predict)
    assert [len(c) for c in calls] == [64, 64, 2]
    assert len(records) == len(dones) == 130
    assert [d[1] for d in dones] == list(range(130))
    assert all(d[4] == 1 for d in dones)
    # every done in one predict chunk shares that chunk's t0/worker_s
    assert len({d[2] for d in dones[:64]}) == 1
    assert dones[64][2] != dones[0][2]


def test_sweep_needs_fanout_and_one_shard() -> None:
    cfg = {
        "name": "x",
        "rerankers": {
            "prod": {"source": "production"},
            "kev": {"source": "kev", "method": "noul_query_in_state", "sweep": "default"},
        },
    }
    with pytest.raises(ValidationError, match="sweep needs fanout"):
        Experiment.model_validate(cfg)
    cfg["rerankers"]["kev"]["fanout"] = True
    cfg["rerankers"]["kev"]["shards"] = 4
    with pytest.raises(ValidationError, match="shards: 1"):
        Experiment.model_validate(cfg)


def test_default_sweep_grid() -> None:
    exp = Experiment.model_validate(
        {
            "name": "x",
            "rerankers": {
                "prod": {"source": "production"},
                "kev": {
                    "source": "kev",
                    "method": "noul_query_in_state",
                    "max_items": 12,
                    "fanout": True,
                    "sweep": "default",
                },
            },
        }
    )
    run = exp.scoring_run("e", "kev", exp.rerankers["kev"])
    assert [(p.requests, p.children) for p in run.sweep] == [
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (1, 16),
        (1, 32),
        (1, 64),
        (1, 128),
        (2, 12),
        (4, 12),
        (8, 12),
    ]


def test_sweep_stats_records_a_failed_point_and_keeps_the_next_point_clean() -> None:
    rows = [
        _rep(1.0),
        {
            "requests": 1,
            "children": 32,
            "rep": 3,
            "case_id": "case_1",
            "children_sent": 32,
            "warmup": False,
            "error": "case_000 batch 0: ValueError: schema requires too many tokens",
        },
        _rep(2.0, requests=1, children=64),
    ]
    stats = {pt["children"]: pt for pt in sweep_stats(rows)}
    assert stats[32]["failed"] and stats[32]["p50_s"] is None
    assert "too many tokens" in stats[32]["error"]
    assert stats[32]["p50_ratio"] is None and stats[32]["children_per_s"] is None
    assert not stats[64]["failed"] and stats[64]["p50_ratio"] == pytest.approx(2.0)
    # a failed baseline leaves every ratio null
    stats = {pt["children"]: pt for pt in sweep_stats([{**rows[1], "children": 1}])}
    assert stats[1]["failed"] and stats[1]["p50_ratio"] is None

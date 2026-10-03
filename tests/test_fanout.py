"""Fan-out latency runs: worker assignment, dispatcher timing overwrite, sweep stats, validation."""

import pytest
from pydantic import ValidationError

from jev_tracker.experiment import Experiment, apply_dispatch, sweep_stats
from jev_tracker.modal_app import fanout_max_items, fanout_plan
from jev_tracker.systemone import RawRecord, Usage


def test_fanout_assigns_request_j_to_worker_j_mod_shards() -> None:
    assert fanout_plan(7, 3) == [[0, 3, 6], [1, 4], [2, 5]]
    assert fanout_plan(2, 4) == [[0], [1], [], []]


def test_fanout_max_items_gives_one_request_per_worker() -> None:
    # 30 children on a 4-GPU pool: batches of 8 -> 4 requests on 4 distinct workers.
    m = fanout_max_items(30, 4, 12)
    assert m == 8
    assert (30 + m - 1) // m <= 4
    assert fanout_max_items(30, 4, 5) == 5  # run.max_items caps the request size
    assert fanout_max_items(30, 4, None) == 8


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


def test_laya_fanout_is_rejected() -> None:
    cfg = {
        "name": "x",
        "rerankers": {
            "prod": {"source": "production"},
            "laya": {"source": "laya", "method": "noul_query_in_state", "fanout": True},
        },
    }
    with pytest.raises(ValidationError, match="laya cannot fan out"):
        Experiment.model_validate(cfg)


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

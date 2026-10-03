"""R-6: site/public/data/rows.json is regenerated from the committed experiments and shows the
attached kev_cost_report's numbers unchanged."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from jev_tracker.evaluation_queue import Queue, QueueItem
from jev_tracker.experiment import EXPERIMENTS_DIR
from jev_tracker.server import (
    check_skip,
    decided,
    parse_decision,
    parse_request,
    request_path,
    suggestions,
)
from jev_tracker.site_data import (
    API_TIMING,
    OUT,
    REGISTRY,
    RUNTIMES,
    TLDR,
    build,
    read_api_timing,
    updated,
)

DATA = build(
    yaml.safe_load(REGISTRY.read_text()),
    EXPERIMENTS_DIR,
    read_api_timing(API_TIMING),
    TLDR.read_text(),
)


def _row(experiment: str, reranker: str) -> dict:
    return next(
        r for r in DATA["rows"] if (r["experiment"], r["reranker"]) == (experiment, reranker)
    )


# Cells copied from kev_cost_report(4).html: (experiment, reranker) -> rendered values.
REPORT_CELLS = [
    (
        "2026_09_29_08_47_55_good-midge",
        "prod",
        "0.711 0.726 0.766 0.837",
        "4.73 0.0631 63.0",
        "6.21 4.07 23.07",
    ),
    (
        "2026_09_29_08_47_55_good-midge",
        "jev_noul",
        "0.878 0.893 0.925 0.951",
        "0.35 0.0047 4.7",
        "0.21 0.20 0.32",
    ),
    (
        "2026_09_29_08_47_55_good-midge",
        "kev27b_noul",
        "0.875 0.893 0.919 0.950",
        "1.31 0.0175 17.5",
        "17.34 12.56 41.66",
    ),
    (
        "2026_10_02_00_08_39_safe-joey",
        "kev27b_noul",
        "0.875 0.893 0.919 0.950",
        "1.30 0.0174 17.4",
        "17.14 12.47 43.11",
    ),
]


@pytest.mark.parametrize("experiment,reranker,kept,cost,latency", REPORT_CELLS)
def test_R6_rows_show_the_reports_numbers(experiment, reranker, kept, cost, latency) -> None:
    r = _row(experiment, reranker)
    assert " ".join(f"{r['kept_mass'][k]:.3f}" for k in DATA["ks"]) == kept
    c, lat = r["cost"], r["latency"]
    assert f"{c['warm_usd']:.2f} {c['usd_per_query']:.4f} {c['usd_per_1k']:.1f}" == cost
    if r["sources"]["latency"].startswith(f"data/experiments/{experiment}/"):
        assert f"{lat['s_per_query']:.2f} {lat['p50_s']:.2f} {lat['p95_s']:.2f}" == latency


def test_R6_no_deprecated_or_pending_row_reaches_rows_json() -> None:
    registry = yaml.safe_load(REGISTRY.read_text())
    rows = registry["rows"]
    deprecated = {(r["experiment"], r["reranker"]) for r in rows if "deprecated" in r}
    pending = {(r["experiment"], r["reranker"]) for r in rows if "pending_fanout" in r}
    assert deprecated and pending
    shown = {(r["experiment"], r["reranker"]) for r in DATA["rows"]}
    assert not (deprecated | pending) & shown
    assert set(DATA["experiments"]) == {r["experiment"] for r in DATA["rows"]}


def test_R6_committed_rows_json_is_the_generated_one() -> None:
    assert json.loads(OUT.read_text()) == DATA


def test_R6_every_shown_number_names_its_source() -> None:
    for r in DATA["rows"]:
        if r["kept_mass"] is not None:
            assert r["sources"]["kept_mass"], r["reranker"]
        if r["cost"] is not None:
            assert r["sources"]["cost"] and r["sources"]["latency"], r["reranker"]


# R-7: the server takes the box's text and files it under requests/ (the GitHub write is not tested).
def test_R6_every_row_names_a_known_runtime() -> None:
    assert {r["runtime"] for r in DATA["rows"]} <= RUNTIMES
    registry = yaml.safe_load(REGISTRY.read_text())
    registry["rows"][0]["runtime"] = "TensorFlow"
    with pytest.raises(ValueError, match="TensorFlow"):
        build(registry, EXPERIMENTS_DIR, read_api_timing(API_TIMING))


def test_R7_parse_request_accepts_any_text() -> None:
    assert (
        parse_request(b'{"text": " https://huggingface.co/org/model "}')
        == "https://huggingface.co/org/model"
    )
    assert parse_request(b'{"text": "try the new Kev 30B"}') == "try the new Kev 30B"


@pytest.mark.parametrize(
    "body, message",
    [
        (b"not json", "body is not JSON"),
        (b'{"url": "https://x.y"}', "expected {'text'"),
        (b'{"text": "   "}', "expected {'text'"),
    ],
)
def test_R7_parse_request_rejects(body: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        parse_request(body)


def test_R7_request_path_is_dated_and_slugged() -> None:
    now = datetime(2026, 10, 2, 7, 5, 9, tzinfo=UTC)
    path = request_path("https://huggingface.co/Org/Model-Name?x=1", now)
    assert path == "requests/2026-10-02/070509_huggingface_co_org_model_name_x_1.json"
    assert (
        request_path("try the new Kev 30B", now)
        == "requests/2026-10-02/070509_try_the_new_kev_30b.json"
    )


def test_R6_updated_and_cards() -> None:
    assert (
        updated(["2026_10_02_03_29_11_above-dog", "2026_09_28_01_00_00_old-cat"])
        == "2026-10-01 20:29 PDT"
    )
    with pytest.raises(ValueError, match="no experiment id carries a timestamp"):
        updated(["nostamp"])
    data = json.loads(OUT.read_text())
    assert [c["label"] for c in data["cards"]] == [
        "best quality (mean kept-mass)",
        "cheapest ($ / 1k queries)",
        "fastest (s / query)",
    ]
    assert all("Jev" in c["detail"] for c in data["cards"])
    assert data["updated"] == updated(list(data["experiments"]))


def test_R7_parse_decision() -> None:
    assert parse_decision(b'{"config": "configs/clef9b.yaml", "decision": "approve"}') == (
        Path("configs/clef9b.yaml"),
        "approve",
    )
    for body in (
        b"nope",
        b'{"config": "", "decision": "approve"}',
        b'{"config": "x", "decision": "maybe"}',
    ):
        with pytest.raises(ValueError):
            parse_decision(body)


def test_R7_skip_needs_the_phrase_and_the_password() -> None:
    ok = b'{"config": "configs/a.yaml", "decision": "reject", "phrase": "Skip this Run", "password": "pw"}'
    check_skip(ok, "pw")
    with pytest.raises(ValueError, match="type 'Skip this Run'"):
        check_skip(ok.replace(b"Skip this Run", b"skip this run"), "pw")
    with pytest.raises(ValueError):
        check_skip(b'{"config": "configs/a.yaml", "decision": "reject"}', "pw")
    with pytest.raises(PermissionError, match="wrong password"):
        check_skip(ok.replace(b'"pw"', b'"PW"'), "pw")
    with pytest.raises(PermissionError, match="wrong password"):
        check_skip(ok.replace(b', "password": "pw"', b""), "pw")
    with pytest.raises(PermissionError, match="QUEUE_SKIP_PASSWORD is not set"):
        check_skip(ok, None)


def test_R7_decided_moves_only_proposed_items() -> None:
    now = datetime(2026, 10, 2, 13, 0, tzinfo=UTC)
    queue = Queue(
        items=[
            QueueItem(
                config=Path("configs/a.yaml"),
                label="a",
                source="hf",
                url="",
                status="proposed",
                queued_at=now,
            ),
            QueueItem(
                config=Path("configs/b.yaml"),
                label="b",
                source="hf",
                url="",
                status="running",
                queued_at=now,
            ),
        ]
    )
    assert [i.status for i in decided(queue, Path("configs/a.yaml"), "approve", now).items] == [
        "queued",
        "running",
    ]
    assert [i.label for i in decided(queue, Path("configs/a.yaml"), "reject", now).items] == ["b"]
    assert decided(queue, Path("configs/b.yaml"), "reject", now) == queue


def _file(name: str, text: str, submitted_at: str) -> dict:
    return {
        "name": name,
        "object": {"text": json.dumps({"text": text, "submitted_at": submitted_at})},
    }


def test_R7_suggestions_lists_every_submission_newest_first() -> None:
    payload = {
        "data": {
            "repository": {
                "object": {
                    "entries": [
                        {
                            "name": "2026-10-01",
                            "object": {
                                "entries": [_file("090000_a.json", "a", "2026-10-01T09:00:00Z")]
                            },
                        },
                        {
                            "name": "2026-10-02",
                            "object": {
                                "entries": [
                                    _file("010000_b.json", "b", "2026-10-02T01:00:00Z"),
                                    _file(
                                        "120000_c.json",
                                        "try the new Kev 30B",
                                        "2026-10-02T12:00:00Z",
                                    ),
                                ]
                            },
                        },
                        {"name": "README.md", "object": {}},
                    ]
                }
            }
        }
    }
    got = suggestions(payload)
    assert [s.text for s in got] == ["try the new Kev 30B", "b", "a"]
    assert got[0].path == "requests/2026-10-02/120000_c.json"
    assert suggestions({"data": {"repository": {"object": None}}}) == []


def test_R6_latency_from_overrides_latency_but_not_cost() -> None:
    registry = yaml.safe_load(REGISTRY.read_text())
    donor = _row("2026_10_02_00_08_39_safe-joey", "kev27b_noul")
    row = next(
        r
        for r in registry["rows"]
        if (r["experiment"], r["reranker"]) == ("2026_10_03_00_31_36_proper-bee", "kev4b_noul")
    )
    donor_reg = next(
        r
        for r in registry["rows"]
        if (r["experiment"], r["reranker"]) == (donor["experiment"], donor["reranker"])
    )
    row["latency_from"] = donor_reg.get(
        "latency_from", {"experiment": donor["experiment"], "reranker": donor["reranker"]}
    )
    data = build(registry, EXPERIMENTS_DIR, read_api_timing(API_TIMING))
    built = next(
        r
        for r in data["rows"]
        if (r["experiment"], r["reranker"]) == (row["experiment"], row["reranker"])
    )
    assert built["latency"] == donor["latency"]
    assert built["sources"]["latency"] == donor["sources"]["latency"]
    own = _row("2026_10_03_00_31_36_proper-bee", "kev4b_noul")
    assert built["cost"] == own["cost"] and built["kept_mass"] == own["kept_mass"]

"""R-6: site/public/data/rows.json is regenerated from the committed experiments and shows the
attached kev_cost_report's numbers unchanged."""

import json
import re
from datetime import UTC, datetime

import pytest
import yaml

from jev_tracker.experiment import EXPERIMENTS_DIR
from jev_tracker.server import parse_request, request_path
from jev_tracker.site_data import API_TIMING, OUT, REGISTRY, TLDR, build, read_api_timing, updated

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
        "465.4 6.21 1.72",
    ),
    (
        "2026_09_29_08_47_55_good-midge",
        "jev_noul",
        "0.878 0.893 0.925 0.951",
        "0.35 0.0047 4.7",
        "15.5 0.21 0.06",
    ),
    (
        "2026_09_28_17_56_26_actual-clam",
        "kev4b_noul",
        "0.724 0.780 0.835 0.884",
        "0.61 0.0081 8.1",
        "1127 15.03 4.17",
    ),
    (
        "2026_09_28_22_15_35_pretty-sawfly",
        "kev27b_noul",
        "0.877 0.893 0.919 0.950",
        "1.40 0.0187 18.7",
        "906 12.08 3.36",
    ),
]


@pytest.mark.parametrize("experiment,reranker,kept,cost,latency", REPORT_CELLS)
def test_R6_rows_show_the_reports_numbers(experiment, reranker, kept, cost, latency) -> None:
    r = _row(experiment, reranker)
    assert " ".join(f"{r['kept_mass'][k]:.3f}" for k in DATA["ks"]) == kept
    c, lat = r["cost"], r["latency"]
    assert f"{c['warm_usd']:.2f} {c['usd_per_query']:.4f} {c['usd_per_1k']:.1f}" == cost
    assert f"{lat['run_s']:g} {lat['s_per_query']:.2f} {lat['h_per_1k']:.2f}" == latency


def test_R6_committed_rows_json_is_the_generated_one() -> None:
    assert json.loads(OUT.read_text()) == DATA


def test_R6_every_shown_number_names_its_source() -> None:
    for r in DATA["rows"]:
        if r["kept_mass"] is not None:
            assert r["sources"]["kept_mass"], r["reranker"]
        if r["cost"] is not None:
            assert r["sources"]["cost"] and r["sources"]["latency"], r["reranker"]


# R-7: the server takes the box's text and files it under requests/ (the GitHub write is not tested).
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

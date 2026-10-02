"""R-6: site/public/data/rows.json is regenerated from the committed experiments and shows the
attached kev_cost_report's numbers unchanged."""

import json

import pytest
import yaml

from jev_tracker.experiment import EXPERIMENTS_DIR, REPO_DIR, load_config
from jev_tracker.site_data import API_TIMING, OUT, REGISTRY, build, read_api_timing

DATA = build(yaml.safe_load(REGISTRY.read_text()), EXPERIMENTS_DIR, read_api_timing(API_TIMING))


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


# R-7 lives in the browser (site/src/evaluate_config.ts); this pins the two things Python can check.
EVALUATE_TS = REPO_DIR / "site" / "src" / "evaluate_config.ts"

# What the form emits for source=kev, name=newkev, both methods: keep in step with evaluate_config.ts.
FORM_YAML = """name: newkev
cases: null
k: {start: 50, stop: 200, step: 10}
rerankers:
  prod:
    source: production
  jev_noul:
    source: answers
    method: noul_query_in_state
    path: data/jev/noul_query_in_state.json.gz
  jev_score:
    source: answers
    method: score_query_in_question
    path: data/jev/score_query_in_question.json.gz
  newkev_noul:
    source: kev
    model: org/new-kev
    max_items: 12
    max_chars: 12000
    shards: 3
    concurrency: 16
    method: noul_query_in_state
  newkev_score:
    source: kev
    model: org/new-kev
    max_items: 12
    max_chars: 12000
    shards: 3
    concurrency: 16
    method: score_query_in_question
"""


def test_R7_issue_url(tmp_path) -> None:
    src = EVALUATE_TS.read_text()
    assert "/issues/new?labels=evaluate" in src
    assert "token" not in src.lower()
    for line in FORM_YAML.splitlines():
        if line.startswith("    ") and "org/new-kev" not in line and "method:" not in line:
            assert line in src, line
    path = tmp_path / "newkev.yaml"
    path.write_text(FORM_YAML)
    exp = load_config(path)
    assert set(exp.rerankers) == {"prod", "jev_noul", "jev_score", "newkev_noul", "newkev_score"}

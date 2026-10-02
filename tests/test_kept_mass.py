"""R-4 (offline half): the ported metric reproduces the committed numbers.

prod + Jev from the committed answers must give the report's kept-mass@k exactly, and every
committed experiment's raw answers must re-score to its committed kept_mass_<id>.json.
"""

import json
from pathlib import Path

import pytest

from jev_tracker.experiment import EXPERIMENTS_DIR, AnswersSource, Paths, load_config
from jev_tracker.kept_mass import evaluate, tables
from jev_tracker.methods import METHODS
from jev_tracker.rerankers import ProductionReranker, Reranker, ScoredReranker

REPO = Path(__file__).resolve().parents[1]
KS = [50, 100, 150, 200]
# docs/kev_cost_report.html, Table 3 (3 decimals); prod and Jev are frozen inputs.
REPORT = {
    "prod": (0.711, 0.726, 0.766, 0.837),
    "jev_noul": (0.878, 0.893, 0.925, 0.951),
    "jev_score": (0.893, 0.912, 0.938, 0.961),
    "random": (0.443, 0.542, 0.636, 0.749),
    "oracle": (1.0, 1.0, 1.0, 1.0),
}


def test_R4_prod_and_jev_reproduce_the_report() -> None:
    rerankers: list[Reranker] = [
        ProductionReranker("prod"),
        ScoredReranker(
            "jev_noul",
            METHODS["noul_query_in_state"],
            REPO / "data/jev/noul_query_in_state.json.gz",
        ),
        ScoredReranker(
            "jev_score",
            METHODS["score_query_in_question"],
            REPO / "data/jev/score_query_in_question.json.gz",
        ),
    ]
    per_case, n_cases = evaluate(rerankers, KS, None)
    assert n_cases == 75
    means, wins = tables(per_case, KS)
    for name, expected in REPORT.items():
        assert tuple(round(means[name][k], 3) for k in KS) == expected, name
    assert all(wins["jev_score"][k] > 0.9 for k in KS)


def _rescorable(id: str) -> bool:
    """An experiment whose config the ported harness still reads (no dropped source kinds)."""
    try:
        load_config(Paths(id).config)
    except Exception:  # noqa: BLE001 - vllm / fp8 / clm configs: data, not inputs to this harness
        return False
    return True


COMMITTED = sorted(p.name for p in EXPERIMENTS_DIR.iterdir() if p.is_dir())


@pytest.mark.parametrize("id", [e for e in COMMITTED if _rescorable(e)])
def test_R4_committed_raw_answers_rescore_to_the_committed_table(id: str) -> None:
    p = Paths(id)
    exp = load_config(p.config)
    committed = json.loads((p.dir / f"kept_mass_{id}.json").read_text())["kept_mass"]
    rerankers: list[Reranker] = []
    for name, src in exp.rerankers.items():
        if src.source == "production":
            rerankers.append(ProductionReranker(name))
        elif p.raw(name).exists():  # a reranker whose shards all failed has no raw file
            method = src.method if isinstance(src, AnswersSource) else src.method
            rerankers.append(ScoredReranker(name, METHODS[method], p.raw(name)))
    per_case, _ = evaluate(rerankers, exp.k.ks, exp.cases)
    means, _ = tables(per_case, exp.k.ks)
    for name, row in means.items():
        for k in exp.k.ks:
            assert abs(row[k] - committed[name][str(k)]) < 1e-9, (id, name, k)

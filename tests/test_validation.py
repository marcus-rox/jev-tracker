"""R-4: the ported harness, run on Modal, reproduces rox-research's committed kept-mass.

Each pair is (jev-tracker rerun, rox-research reference) for the same model, method and
serving. Tolerance is Kev's measured rerun noise (|delta| <= 0.002 at every reported k).
"""

import json

import pytest

from jev_tracker.experiment import Paths

TOLERANCE = 0.002
KS = ("50", "100", "150", "200")

RERUNS = [
    ("2026_10_01_23_53_13_keen-emu", "kev4b_noul", "2026_09_28_17_56_26_actual-clam", "kev4b_noul"),
    (
        "2026_10_01_23_53_13_keen-emu",
        "kev4b_score",
        "2026_09_28_17_56_26_actual-clam",
        "kev4b_score",
    ),
    (
        "2026_10_01_23_53_20_frank-moray",
        "laya_noul",
        "2026_09_29_21_58_31_top-ram",
        "laya421m_noul",
    ),
    (
        "2026_10_01_23_53_20_frank-moray",
        "laya_score",
        "2026_09_29_21_58_31_top-ram",
        "laya421m_score",
    ),
    (
        "2026_10_01_23_53_25_fit-macaw",
        "kev27b_noul",
        "2026_09_28_22_15_35_pretty-sawfly",
        "kev27b_noul",
    ),
    (
        "2026_10_01_23_53_25_fit-macaw",
        "kev27b_score",
        "2026_09_28_22_15_35_pretty-sawfly",
        "kev27b_score",
    ),
]


def _kept_mass(experiment: str, reranker: str) -> tuple[int, dict[str, float]]:
    report = json.loads(Paths(experiment).report_base.with_suffix(".json").read_text())
    return report["n_cases"], report["kept_mass"][reranker]


@pytest.mark.parametrize("rerun,row,reference,ref_row", RERUNS)
def test_R4_rerun_matches_rox_research_within_noise(rerun, row, reference, ref_row):
    n, got = _kept_mass(rerun, row)
    n_ref, want = _kept_mass(reference, ref_row)
    assert n == n_ref == 75
    deltas = {k: round(got[k] - want[k], 4) for k in KS}
    assert all(abs(d) <= TOLERANCE for d in deltas.values()), deltas

"""ONNX task: R-1 `source: ollaya` configuration, R-2 pull progress + response shape, R-3 smoke
config. No GPU and no network: the Modal function itself is exercised by the smoke run."""

import json
from pathlib import Path

import pytest
from tqdm import tqdm

from jev_tracker import modal_app
from jev_tracker.experiment import OllayaSource, load_config
from jev_tracker.methods import METHODS, Item, record
from jev_tracker.systemone import SystemOneResponse

SMOKE = Path("configs/kev9b_ollaya_smoke.yaml")
RUN = "kev9b_onnx_noul"


def test_R3_smoke_config_loads() -> None:
    exp = load_config(SMOKE)
    src = exp.rerankers[RUN]
    assert isinstance(src, OllayaSource)
    assert exp.cases == 2
    assert src.model == "kev:9b" and src.gpu == "H100"
    assert {"prod", "jev_noul", "jev_score"} <= set(exp.rerankers)


def test_R1_dispatch_goes_to_the_ollaya_scorer_on_the_configured_gpu() -> None:
    exp = load_config(SMOKE)
    run = exp.scoring_run("x", RUN, exp.rerankers[RUN])
    assert run.engine == "ollaya"
    assert modal_app.SCORERS[run.engine] is modal_app.score_cases_ollaya
    assert run.gpu_type == "H100"
    assert run.model_copy(update={"gpu": None}).gpu_type == modal_app.DEFAULT_OLLAYA_GPU
    assert run.name.startswith("kev:9b_noul_query_in_state")


# /api/pull's NDJSON (docs/api.md §7.6): one line per progress step, `success` last.
PULL_LINES = [
    {"status": "pulling manifest"},
    {"status": "pulling 4396176", "digest": "sha256:4396176abcdef", "total": 100, "completed": 0},
    {"status": "pulling 4396176", "digest": "sha256:4396176abcdef", "total": 100, "completed": 60},
    {"status": "pulling 4396176", "digest": "sha256:4396176abcdef", "total": 100, "completed": 100},
    {"status": "pulling 862bf7b", "digest": "sha256:862bf7b123456", "total": 50, "completed": 50},
    {"status": "verifying sha256 digest"},
    {"status": "writing manifest"},
    {"status": "success"},
]


def test_R2_pull_progress_drives_one_bar_per_blob_to_its_total() -> None:
    bars: dict[str, tqdm] = {}
    modal_app.pull_progress([json.dumps(line) for line in PULL_LINES] + [""], bars)
    assert {d: (b.n, b.total) for d, b in bars.items()} == {
        "sha256:4396176abcdef": (100, 100),
        "sha256:862bf7b123456": (50, 50),
    }


def test_R2_pull_stream_cut_before_success_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="without success"):
        modal_app.pull_progress([json.dumps(line) for line in PULL_LINES[:3]], {})
    with pytest.raises(RuntimeError, match="DIGEST_MISMATCH"):
        modal_app.pull_progress([json.dumps({"error": "DIGEST_MISMATCH: x"})], {})


# Ollaya's /v1/systemone answer (docs/api.md §8.1): wire-identical to TypeSafe's.
OLLAYA_RESPONSE = {
    "model": "kev:9b",
    "answers": {
        "i0": {"type": "noul", "noul": 0.9321},
        "i1": {"type": "noul", "noul": 0.0417},
    },
    "usage": {"input_tokens": 211, "output_tokens": 0},
}
BATCH = [
    Item(parent_index=0, child_index=0, text="Paris is the capital of France."),
    Item(parent_index=0, child_index=1, text="Cats purr."),
]


def test_R2_ollaya_response_validates_and_records() -> None:
    resp = SystemOneResponse.model_validate(OLLAYA_RESPONSE)
    rec = record(METHODS["noul_query_in_state"], "case-1", 0, BATCH, resp, latency_s=0.5)
    assert rec.model == "kev:9b"
    assert [s.child_index for s in rec.scores] == [0, 1]
    assert rec.usage.input_tokens == 211

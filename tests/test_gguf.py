"""GGUF task (docs/GGUF_SPEC.html): R-1 configuration, R-2 response shape, R-3 smoke config.

No GPU and no network: the Modal function itself is exercised by the smoke run (R-2 on Modal)."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from jev_tracker import modal_app
from jev_tracker.experiment import Experiment, GgufSource, load_config
from jev_tracker.methods import METHODS, Item, record
from jev_tracker.systemone import SystemOneResponse

SMOKE = Path("configs/kev9b_gguf_smoke.yaml")
RUN = "kev9b_q4_noul"


def test_R3_smoke_config_loads() -> None:
    exp = load_config(SMOKE)
    src = exp.rerankers[RUN]
    assert isinstance(src, GgufSource)
    assert exp.cases == 2
    assert src.model == "ggml-org/Kev-9B-GGUF@ec2bbfe6620218aee2e01cc93bb78dee2a96ed58"
    assert src.file == "Kev-9B-Q4_K_M.gguf"
    assert src.gpu == "L40S"
    assert {"prod", "jev_noul", "jev_score"} <= set(exp.rerankers)


def test_R1_switch_quantization_changes_only_the_file() -> None:
    raw = yaml.safe_load(SMOKE.read_text())
    q4 = Experiment.model_validate(raw).scoring_run(
        "x", RUN, Experiment.model_validate(raw).rerankers[RUN]
    )
    raw["rerankers"][RUN]["file"] = "Kev-9B-Q8_0.gguf"
    q8 = Experiment.model_validate(raw).scoring_run(
        "x", RUN, Experiment.model_validate(raw).rerankers[RUN]
    )
    assert q4.engine == q8.engine == "gguf"
    assert q4.file == "Kev-9B-Q4_K_M.gguf" and q8.file == "Kev-9B-Q8_0.gguf"
    assert q4.name.startswith("Kev-9B-Q4_K_M") and q8.name.startswith("Kev-9B-Q8_0")
    assert q4.model_dump(exclude={"file"}) == q8.model_dump(exclude={"file"})


def test_R1_missing_file_is_a_config_error_naming_file() -> None:
    raw = yaml.safe_load(SMOKE.read_text())
    del raw["rerankers"][RUN]["file"]
    with pytest.raises(ValidationError, match="file"):
        Experiment.model_validate(raw)


def test_R1_dispatch_goes_to_the_gguf_scorer_on_the_configured_gpu() -> None:
    exp = load_config(SMOKE)
    run = exp.scoring_run("x", RUN, exp.rerankers[RUN])
    assert modal_app.SCORERS[run.engine] is modal_app.score_cases_gguf
    assert run.gpu_type == "L40S"
    bare = run.model_copy(update={"gpu": None})
    assert bare.gpu_type == modal_app.DEFAULT_GGUF_GPU
    assert modal_app.MEMORY_MB_FOR.get(run.repo) is None


# llama-server's /v1/systemone answer for a noul_query_in_state request over two items, in the
# shape its README documents (type per answer, usage with output_tokens 0).
LLAMA_SERVER_RESPONSE = {
    "model": "Kev-9B-Q4_K_M.gguf",
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


def test_R2_llama_server_response_validates_and_records() -> None:
    resp = SystemOneResponse.model_validate(LLAMA_SERVER_RESPONSE)
    rec = record(METHODS["noul_query_in_state"], "case-1", 0, BATCH, resp, latency_s=0.5)
    assert rec.model == "Kev-9B-Q4_K_M.gguf"
    assert [s.child_index for s in rec.scores] == [0, 1]
    assert rec.usage.input_tokens == 211

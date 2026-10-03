"""R-3: every experiment is fully recorded under data/experiments/<id>/ and labelled in
data/registry.yaml; every config in configs/ loads; committed latency re-derives from raw answers."""

import json
from pathlib import Path

import pytest
import yaml

from jev_tracker.experiment import CONFIGS_DIR, EXPERIMENTS_DIR, ApiSource, Paths, load_config
from jev_tracker.metrics import query_wall_s
from jev_tracker.rerankers import read_raw

REPO = Path(__file__).resolve().parents[1]
REGISTRY = yaml.safe_load((REPO / "data/registry.yaml").read_text())
COMMITTED = sorted(p.name for p in EXPERIMENTS_DIR.iterdir() if p.is_dir())
# An experiment whose every scored arm failed has no table; its registry rows all say "failed".
FAILED = {
    e
    for e in COMMITTED
    if (rows := [r for r in REGISTRY["rows"] if r["experiment"] == e])
    and all(r.get("blank") == "failed" for r in rows)
}


def _latency_only(id: str) -> bool:
    """A fan-out latency run or batch-size sweep: no production arm, its rows serve `latency_from`."""
    exp = load_config(Paths(id).config)
    return all(
        getattr(s, "fanout", False) or getattr(s, "sweep", None) is not None
        for s in exp.rerankers.values()
    )


def _kept_mass(id: str) -> dict:
    return json.loads((Paths(id).dir / f"kept_mass_{id}.json").read_text())


@pytest.mark.parametrize("config", sorted(CONFIGS_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_R1_every_config_loads_and_names_committed_answers(config: Path) -> None:
    exp = load_config(config)
    assert exp.name == config.stem
    sweep_only = all(getattr(src, "sweep", None) is not None for src in exp.rerankers.values())
    if not sweep_only:
        assert "prod" in exp.rerankers and exp.rerankers["prod"].source == "production"
    for name, src in exp.rerankers.items():
        if src.source == "answers":
            assert (REPO / src.path).exists(), (name, src.path)
        if isinstance(src, ApiSource):
            assert src.url.startswith("https://") and src.secret  # never a key in the config
            run = exp.scoring_run("x", name, src)
            assert run.gpu_type == "API" and run.api is not None and run.api.url == src.url


@pytest.mark.parametrize("id", COMMITTED)
def test_R3_experiment_dir_holds_config_and_kept_mass_under_its_id(id: str) -> None:
    p = Paths(id)
    assert p.config.exists(), id
    if id in FAILED:
        assert p.calls.exists() and not list(p.dir.glob("raw_*")), id
    else:
        km = _kept_mass(id)
        assert set(km) >= {"n_cases", "kept_mass"}
        assert "prod" in km["kept_mass"] or _latency_only(id), id
    for f in p.dir.iterdir():
        assert id in f.name, f  # nothing anonymous: every artifact carries the experiment id
        if f.name.startswith("raw_"):
            assert f.suffixes == [".json", ".gz"], f
    assert id in REGISTRY["experiments"], id
    assert not (p.dir / f"raw_prod_{id}.json.gz").exists()  # production needs no answers


def test_R3_registry_rows_point_at_committed_numbers() -> None:
    rows = REGISTRY["rows"]
    assert len({(r["experiment"], r["reranker"]) for r in rows}) == len(rows)
    for r in rows:
        assert set(r) >= {"experiment", "reranker", "label", "family", "serving", "gpu", "queries"}
        if r["experiment"] in FAILED or "deprecated" in r:
            continue
        km = _kept_mass(r["experiment"])
        assert km["n_cases"] == r["queries"], r
        if r["reranker"] in km["kept_mass"] or r.get("blank") == "failed":
            continue
        # a timing-only arm (no answers): its numbers are in costs_<id>.json
        costs = json.loads(Paths(r["experiment"]).costs.read_text())["rerankers"]
        assert r["reranker"] in costs, r


@pytest.mark.parametrize("id", [e for e in COMMITTED if Paths(e).latency.exists()])
def test_R3_committed_latency_rederives_from_raw_answers(id: str) -> None:
    p = Paths(id)
    committed = json.loads(p.latency.read_text())["rerankers"]
    for name, row in committed.items():
        per = query_wall_s(read_raw(p.raw(name)))
        assert per.keys() == row["per_case_s"].keys(), (id, name)
        for case, s in per.items():
            assert abs(s - row["per_case_s"][case]) < 1e-6, (id, name, case)

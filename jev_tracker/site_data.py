"""R-6: the site's data files, regenerated from data/registry.yaml and data/experiments/.

    python -m jev_tracker.site_data            # writes site/public/data/rows.json

Every number the site shows comes from a committed JSON (or the API timing CSV) named in the
row's `sources`, so the browser can link each cell to where it came from.
"""

import csv
import json
import sys
from pathlib import Path

import yaml

from jev_tracker.experiment import EXPERIMENTS_DIR, REPO_DIR, Paths

REGISTRY = REPO_DIR / "data" / "registry.yaml"
API_TIMING = REPO_DIR / "data" / "timing_summary_prod_jev.csv"
OUT = REPO_DIR / "site" / "public" / "data" / "rows.json"
KS = ("50", "100", "150", "200")
SECONDS_PER_HOUR = 3600


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_DIR))


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _kept_mass(report: dict | None, reranker: str) -> dict[str, float] | None:
    km = (report or {}).get("kept_mass", {}).get(reranker)
    return {k: km[k] for k in KS} if km else None


def _gpu_cost(costs: dict | None, reranker: str, queries: int) -> dict[str, float | None] | None:
    c = (costs or {}).get("rerankers", {}).get(reranker)
    if not c or c.get("warm_gpu_s") is None:
        return None
    return {
        "warm_gpu_s": c["warm_gpu_s"],
        "load_s": c["load_s"],
        "warm_usd": c["warm_usd"],
        "in_function_usd": c["in_function_usd"],
        "usd_per_query": c["warm_usd"] / queries,
        "usd_per_1k": c["warm_usd_per_1k"],
        "peak_gb": c.get("peak_gb"),
    }


def _gpu_latency(cost: dict | None, queries: int) -> dict[str, float] | None:
    """The report's latency columns: the warm GPU-seconds of the run, spread over its queries."""
    if cost is None:
        return None
    s = cost["warm_gpu_s"]
    return {
        "run_s": s,
        "s_per_query": s / queries,
        "h_per_1k": s / queries * 1000 / SECONDS_PER_HOUR,
    }


def _api_cost(row: dict) -> dict[str, float | None] | None:
    if "api_usd_per_run" not in row:
        return None
    run = row["api_usd_per_run"]
    return {
        "warm_gpu_s": None,
        "load_s": None,
        "warm_usd": run,
        "in_function_usd": run,
        "usd_per_query": run / row["queries"],
        "usd_per_1k": row["api_usd_per_1k"],
        "peak_gb": None,
    }


def _api_latency(timing: dict[str, dict[str, str]], reranker: str) -> dict[str, float] | None:
    t = timing.get(reranker)
    if t is None:
        return None
    return {
        "run_s": float(t["total_s_75_queries_sequential"]),
        "s_per_query": float(t["mean_s_per_query"]),
        "h_per_1k": float(t["s_per_1000_queries_extrapolated"]) / SECONDS_PER_HOUR,
    }


API_TIMING_NAMES = {"prod": "production"}  # the CSV calls the production reranker "production"


def read_api_timing(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as f:
        return {r["method"]: r for r in csv.DictReader(f)}


def build(registry: dict, experiments_dir: Path, api_timing: dict[str, dict[str, str]]) -> dict:
    """Pure: one site row per registry row, numbers read from that experiment's committed JSONs."""
    rows = []
    for r in registry["rows"]:
        p = Paths(r["experiment"], experiments_dir)
        report_path = p.report_base.with_suffix(".json")
        report, costs = _json(report_path), _json(p.costs)
        queries = r["queries"]
        cost = _api_cost(r) or _gpu_cost(costs, r["reranker"], queries)
        latency = (
            _api_latency(api_timing, API_TIMING_NAMES.get(r["reranker"], r["reranker"]))
            if "api_usd_per_run" in r
            else _gpu_latency(cost, queries)
        )
        kept = _kept_mass(report, r["reranker"])
        rows.append(
            {
                **{
                    k: r[k] for k in ("experiment", "reranker", "label", "family", "serving", "gpu")
                },
                "buffer": r["buffer"],
                "queries": queries,
                "tables": r["tables"],
                "highlight": r["highlight"],
                "blank": r.get("blank", "-"),
                "kept_mass": kept,
                "mean_kept_mass": sum(kept.values()) / len(KS) if kept else None,
                "cost": cost,
                "latency": latency,
                "sources": {
                    "kept_mass": _rel(report_path) if report_path.exists() else None,
                    "cost": _rel(REGISTRY) if "api_usd_per_run" in r else _rel(p.costs),
                    "latency": _rel(API_TIMING) if "api_usd_per_run" in r else _rel(p.costs),
                    "config": _rel(p.config),
                },
            }
        )
    return {"ks": list(KS), "experiments": registry["experiments"], "rows": rows}


def main(out: Path = OUT) -> None:
    data = build(yaml.safe_load(REGISTRY.read_text()), EXPERIMENTS_DIR, read_api_timing(API_TIMING))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"{len(data['rows'])} rows, {len(data['experiments'])} experiments -> {_rel(out)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)

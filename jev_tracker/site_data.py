"""R-6: the site's data files, regenerated from data/registry.yaml and data/experiments/.

    python -m jev_tracker.site_data            # writes site/public/data/rows.json

Every number the site shows comes from a committed JSON (or the API timing CSV) named in the
row's `sources`, so the browser can link each cell to where it came from.
"""

import csv
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from jev_tracker.experiment import EXPERIMENTS_DIR, REPO_DIR, Paths

REGISTRY = REPO_DIR / "data" / "registry.yaml"
API_TIMING = REPO_DIR / "data" / "timing_summary_prod_jev.csv"
OUT = REPO_DIR / "site" / "public" / "data" / "rows.json"
TLDR = REPO_DIR / "data" / "tldr.md"
BASELINE_FAMILIES = frozenset({"jev", "production", "oracle", "random"})
# How a row's answers were produced; GGUF / ONNX / MLX / Core ML are the exported-weight runtimes
# the harness has no source for yet (docs/JEV_ALTERNATIVES.html).
RUNTIMES = frozenset(
    {"PyTorch", "vLLM", "SGLang", "GGUF (llama.cpp)", "ONNX", "MLX", "Core ML", "hosted API", "-"}
)
BENCHMARK_QUERIES = 75
PACIFIC = ZoneInfo("America/Los_Angeles")
EXPERIMENT_STAMP = re.compile(r"^(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_")
KS = ("50", "100", "150", "200")


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


def _modal_latency(latency: dict | None, reranker: str) -> dict[str, float] | None:
    """Wall clock per query from latency_<id>.json: first request sent to last answer back."""
    t = (latency or {}).get("rerankers", {}).get(reranker)
    if t is None:
        return None
    return {"s_per_query": t["mean_s"], "p50_s": t["p50_s"], "p95_s": t["p95_s"]}


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
        "s_per_query": float(t["mean_s_per_query"]),
        "p50_s": float(t["median_s_per_query"]),
        "p95_s": float(t["p95_s_per_query"]),
    }


API_TIMING_NAMES = {"prod": "production"}  # the CSV calls the production reranker "production"


def read_api_timing(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as f:
        return {r["method"]: r for r in csv.DictReader(f)}


def updated(experiment_ids: list[str]) -> str:
    """When the newest experiment ran, from its id (YYYY_MM_DD_HH_MM_SS_<petname>, UTC), as Pacific text."""
    stamps = [m for m in (EXPERIMENT_STAMP.match(e) for e in experiment_ids) if m]
    if not stamps:
        raise ValueError(f"no experiment id carries a timestamp: {experiment_ids}")
    newest = datetime(*map(int, max(m.groups() for m in stamps)), tzinfo=UTC)
    return f"{newest.astimezone(PACIFIC):%Y-%m-%d %H:%M %Z}"


def _pick(rows: list[dict], value, lowest: bool) -> dict | None:
    scored = [(value(r), r) for r in rows if value(r) is not None]
    if not scored:
        return None
    return (min if lowest else max)(scored, key=lambda t: t[0])[1]


def _card(
    label: str, rows: list[dict], jev: list[dict], value, fmt: str, lowest: bool, unit: str
) -> dict:
    best, ref = _pick(rows, value, lowest), _pick(jev, value, lowest)
    return {
        "label": label,
        "value": format(value(best), fmt) + unit if best else "-",
        "detail": (
            f"{best['label']} · Jev {format(value(ref), fmt)}{unit}" if best and ref else "-"
        ),
    }


def cards(rows: list[dict]) -> list[dict]:
    """Pure: the best Jev alternative on each axis, over the 75-query rows, with Jev's own number.

    A zero cost or latency means the run was never timed, not that it was free.
    """
    full = [r for r in rows if r["queries"] == BENCHMARK_QUERIES]
    open_rows = [r for r in full if r["family"] not in BASELINE_FAMILIES]
    jev = [r for r in full if r["family"] == "jev"]
    return [
        _card(
            "best quality (mean kept-mass)",
            open_rows,
            jev,
            lambda r: r["mean_kept_mass"],
            ".3f",
            False,
            "",
        ),
        _card(
            "cheapest ($ / 1k queries)",
            open_rows,
            jev,
            lambda r: (r["cost"]["usd_per_1k"] or None) if r["cost"] else None,
            ".2f",
            True,
            "",
        ),
        _card(
            "fastest (s / query)",
            open_rows,
            jev,
            lambda r: (r["latency"]["s_per_query"] or None) if r["latency"] else None,
            ".2f",
            True,
            "",
        ),
    ]


def build(
    registry: dict, experiments_dir: Path, api_timing: dict[str, dict[str, str]], tldr: str = ""
) -> dict:
    """Pure: one site row per registry row, numbers read from that experiment's committed JSONs."""
    rows = []
    for r in registry["rows"]:
        if "deprecated" in r:
            continue
        if r["runtime"] not in RUNTIMES:
            raise ValueError(
                f"{r['experiment']}/{r['reranker']}: runtime {r['runtime']!r} not in {sorted(RUNTIMES)}"
            )
        p = Paths(r["experiment"], experiments_dir)
        report_path = p.report_base.with_suffix(".json")
        report, costs, lat = _json(report_path), _json(p.costs), _json(p.latency)
        queries = r["queries"]
        cost = _api_cost(r) or _gpu_cost(costs, r["reranker"], queries)
        latency = (
            _api_latency(api_timing, API_TIMING_NAMES.get(r["reranker"], r["reranker"]))
            if "api_usd_per_run" in r
            else _modal_latency(lat, r["reranker"])
        )
        kept = _kept_mass(report, r["reranker"])
        rows.append(
            {
                **{
                    k: r[k]
                    for k in (
                        "experiment",
                        "reranker",
                        "label",
                        "family",
                        "serving",
                        "runtime",
                        "gpu",
                    )
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
                    "latency": _rel(API_TIMING) if "api_usd_per_run" in r else _rel(p.latency),
                    "config": _rel(p.config),
                },
            }
        )
    shown = {r["experiment"] for r in rows}
    experiments = {k: v for k, v in registry["experiments"].items() if k in shown}
    return {
        "ks": list(KS),
        "experiments": experiments,
        "rows": rows,
        "updated": updated(list(experiments)),
        "cards": cards(rows),
        "tldr": tldr,
    }


def main(out: Path = OUT) -> None:
    tldr = TLDR.read_text() if TLDR.exists() else ""
    data = build(
        yaml.safe_load(REGISTRY.read_text()), EXPERIMENTS_DIR, read_api_timing(API_TIMING), tldr
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"{len(data['rows'])} rows, {len(data['experiments'])} experiments -> {_rel(out)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)

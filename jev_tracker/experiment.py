"""One experiment = one YAML config run once, identified by <id> = YYYY_MM_DD_HH_MM_SS_<petname>.

Everything the experiment produced lives in data/experiments/<id>/ and carries the id in its name:

    config_<id>.yaml                the config as run (copied from configs/<name>.yaml)
    calls_<id>.jsonl                the Modal calls spawned for it (one per GPU shard)
    raw_<reranker>_<id>.json.gz     every reranker's raw System One answers, all cases
    kept_mass_<id>.{md,json,png}    the table, its numbers and the graph
    costs_<id>.{md,json}            GPU seconds and $ per reranker, from the shards' timers
    latency_<id>.{csv,json}         per-query wall clock per reranker

    uv run python -m jev_tracker.experiment run configs/kev4b_vs_jev.yaml   # spawn GPUs
    uv run python -m jev_tracker.experiment status <id>
    uv run python -m jev_tracker.experiment finish <id>   # pull answers, write the report
    uv run python -m jev_tracker.experiment run configs/x.yaml --wait  # run + finish
    uv run python -m jev_tracker.experiment verify <id>                # did Modal really do it?

Rerankers are declared by `source`: `production` (the frozen ranking in the dataset), `answers`
(raw answers already on disk, e.g. Jev's), `kev`, `laya`, `clef`, `matilda`, `autotrust`, `jevany`,
`rsi_jev` or `minicpm_jev`
(scored now, in-process on Modal, sharded over GPUs), `gguf` (one quantized .gguf file served by
llama.cpp on a Modal GPU) or `api` (any hosted model that answers the System One request at a URL,
from a Modal CPU container). Adding a model = a new source here + a producer of RawRecords in
modal_app.py; the metric is untouched.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Annotated, Literal

import petname
import yaml
from pydantic import BaseModel, Field
from tqdm import tqdm

from jev_tracker import modal_app
from jev_tracker.contract import load_cases
from jev_tracker.kept_mass import PROD, evaluate, tables
from jev_tracker.methods import METHODS
from jev_tracker.metrics import CostRow, parse_summary, query_wall_s, shard_costs
from jev_tracker.report import write_report
from jev_tracker.rerankers import (
    ProductionReranker,
    Reranker,
    ScoredReranker,
    read_raw,
    write_raw,
)
from jev_tracker.systemone import RawRecord

REPO_DIR = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_DIR / "configs"
EXPERIMENTS_DIR = REPO_DIR / "data" / "experiments"
STAMP = "%Y_%m_%d_%H_%M_%S"


class ProductionSource(BaseModel):
    model_config = {"frozen": True}

    source: Literal["production"]


class AnswersSource(BaseModel):
    """Raw answers produced earlier (another model run, another experiment); path from the repo root."""

    model_config = {"frozen": True}

    source: Literal["answers"]
    method: str
    path: Path


class KevSource(BaseModel):
    model_config = {"frozen": True}

    source: Literal["kev"]
    method: str
    model: str = "jaredpalmer/kev-4b"  # any Hugging Face Kev checkpoint
    max_items: int | None = 25  # children per System One request (Kev's state limit: 8,192 tokens)
    max_chars: int | None = 24_000
    shards: int = 1  # GPUs the cases are dealt over
    concurrency: int = 1  # requests in flight per GPU; Kev's server batches them into one pass
    gpu: str | None = None  # Modal GPU type override (e.g. H100); None = modal_app.GPU_FOR[model]


class LayaSource(BaseModel):
    """Laya: a ModernBERT-large encoder whose head scores one [MASK] per option (laya package).

    Laya reads 512 tokens per question, state included, and cuts the rest off silently, so each
    request holds one child: query + child is about 300 tokens. `forward_batch` of those requests
    share one GPU forward pass (laya_batch); 1 = one request per pass (smoke only)."""

    model_config = {"frozen": True}

    source: Literal["laya"]
    method: Literal[
        "noul_query_in_state",
        "noul_query_in_question",
        "noul_query_in_question_options",
        "score_query_in_question",
    ]
    model: Literal[
        "convaiinnovations/laya",
        "convaiinnovations/laya-multilingual",
        "convaiinnovations/laya-typed-decisions",
    ] = modal_app.LAYA_MODEL
    max_items: Literal[1] = 1
    max_chars: None = None
    shards: int = 1
    concurrency: Literal[1] = 1  # laya 0.3.5 shares one tokenizer that is not thread-safe
    forward_batch: int = Field(64, ge=1)
    gpu: str | None = None


class ClefSource(BaseModel):
    """Cloudflare Clef / Clef-Flash: a Qwen3.5 backbone plus a joint schema head that answers the
    System One request body in-process (the release's `joint_schema_model.systemone`).

    One request per forward pass, so one in flight per GPU; the release reads up to 16,384
    tokens, 12 children / 12,000 characters per request matches configs/kev27b_batched.yaml."""

    model_config = {"frozen": True}

    source: Literal["clef"]
    method: str
    model: Literal["Cloudflare/clef", "Cloudflare/clef-flash"]
    max_items: int | None = 12
    max_chars: int | None = 12_000
    shards: int = 1
    concurrency: Literal[1] = 1
    gpu: str | None = None


class MatildaSource(BaseModel):
    """Maincode MATILDA-jev v1: a 26.1B Qwen3.5 backbone plus a decision readout, answered by the
    release's own runtime (`maincode_jev_serve.decide`, as its /v1/systemone server does).

    The runtime repeats the state once per question, so a request holds one child: query + child,
    one question, one forward pass (as Laya); one in flight per GPU."""

    model_config = {"frozen": True}

    source: Literal["matilda"]
    method: str
    model: Literal["Maincode/matilda-jev-v1"] = modal_app.MATILDA_MODEL
    max_items: Literal[1] = 1
    max_chars: None = None
    shards: int = 1
    concurrency: Literal[1] = 1
    gpu: str | None = None


class AutoTrustSource(BaseModel):
    """AutoTrust JEV-27B: Qwen3.8-27B + LoRA + a 24-slot decision head, asked through the release's
    bare prompt one question at a time (jev_tracker.autotrust, the README's transformers path).

    Each question repeats the whole state, so a request holds one child (as Laya); one in flight."""

    model_config = {"frozen": True}

    source: Literal["autotrust"]
    method: str
    model: Literal["autotrust/JEV-27B"] = modal_app.AUTOTRUST_MODEL
    max_items: Literal[1] = 1
    max_chars: None = None
    shards: int = 1
    concurrency: Literal[1] = 1
    gpu: str | None = None


class JevAnySource(BaseModel):
    """JevAny: a pointer LoRA + head on a Qwen3.5 base, loaded by the JevAny release's own runtime
    (`jevany.JevModel`), which answers the System One request body in-process.

    The runtime packs the state once plus one branch per question into 8,192 tokens (and rejects,
    not truncates, anything longer: such a request is split in two); 12 children / 12,000
    characters per request matches configs/kev27b_batched.yaml. One in flight (it holds a lock)."""

    model_config = {"frozen": True}

    source: Literal["jevany"]
    method: str
    model: Literal["SimpleJev/JevAny-Qwen3.5-4B-LoRA"]
    max_items: int | None = 12
    max_chars: int | None = 12_000
    shards: int = 1
    concurrency: Literal[1] = 1
    gpu: str | None = None


class RsiJevSource(BaseModel):
    """RSI-Jev: a fine-tuned Qwen3.5 tower + option scorer, loaded by the release's `load_release`
    and scored with its `rsijev.evaluate.predict`, one encoded row (state + question) per question.

    Its server does not cut a state (32,768 tokens per question); 12 children / 12,000 characters
    per request matches configs/kev27b_batched.yaml. One in flight per GPU."""

    model_config = {"frozen": True}

    source: Literal["rsi_jev"]
    method: str
    model: Literal["shgao/rsi-jev-v3.0-qwen3.5-2b"]
    max_items: int | None = 12
    max_chars: int | None = 12_000
    shards: int = 1
    concurrency: Literal[1] = 1
    gpu: str | None = None


class MiniCpmJevSource(BaseModel):
    """MiniCPM5-2B-Jev: a LoRA + letter readout on MiniCPM5-2B through the release's
    `MiniCPMSystemOne`, which takes the request body (state once, one branch per question).

    Its server keeps 2,560 state tokens (and trims beyond), so a request holds 8 children /
    8,000 characters. One in flight per GPU."""

    model_config = {"frozen": True}

    source: Literal["minicpm_jev"]
    method: str
    model: Literal["ytbai/MiniCPM5-2B-Jev"]
    max_items: int | None = 8
    max_chars: int | None = 8_000
    shards: int = 1
    concurrency: Literal[1] = 1
    gpu: str | None = None


class ApiSource(BaseModel):
    """Any hosted model that answers the System One request body at `url` (Jev, Liquid d1, ...).

    The bearer key is read inside the Modal container from the Modal Secret `secret` (variable
    `key_env`), so no key ever reaches the repo. `max_items` / `max_chars` are the endpoint's
    questions-per-request and tokens-per-question limits in disguise: d1 takes 128 questions and
    32,768 tokens per question, 110,000 characters of children stays under it."""

    model_config = {"frozen": True}

    source: Literal["api"]
    method: str
    url: str
    model: str
    secret: str | None = None
    key_env: str = "API_KEY"
    max_items: int | None = 25
    max_chars: int | None = 24_000
    shards: int = 1
    concurrency: int = 8  # requests in flight per container


class GgufSource(BaseModel):
    """A quantized decision model: one `.gguf` file of a Hub repo, served by llama.cpp's
    /v1/systemone on a Modal GPU. `model` is `repo@revision` (e.g. ggml-org/Kev-9B-GGUF@<sha>),
    `file` the file inside it (Kev-9B-Q4_K_M.gguf, Kev-9B-Q8_0.gguf, ...)."""

    model_config = {"frozen": True}

    source: Literal["gguf"]
    method: str
    model: str
    file: str
    max_items: int | None = 25
    max_chars: int | None = 24_000
    shards: int = 1
    concurrency: int = 8  # llama-server slots = requests in flight per container
    gpu: str | None = None  # None = modal_app.DEFAULT_GGUF_GPU


ModelSource = (
    KevSource
    | LayaSource
    | ClefSource
    | MatildaSource
    | AutoTrustSource
    | JevAnySource
    | RsiJevSource
    | MiniCpmJevSource
    | ApiSource
    | GgufSource
)
Source = Annotated[ProductionSource | AnswersSource | ModelSource, Field(discriminator="source")]


class KGrid(BaseModel):
    model_config = {"frozen": True}

    start: int = 50
    stop: int = 200
    step: int = 10

    @property
    def ks(self) -> list[int]:
        return list(range(self.start, self.stop + 1, self.step))


class Experiment(BaseModel):
    model_config = {"frozen": True}

    name: str
    cases: int | None = None  # first N cases only (smoke); None = all 75
    k: KGrid = KGrid()
    rerankers: dict[str, Source] = Field(min_length=1)  # table rows, in order; PROD for win-rates

    def scoring_run(
        self, id: str, name: str, src: ModelSource, config: str | None = None
    ) -> modal_app.ScoringRun:
        return modal_app.ScoringRun(
            config=config,
            experiment=id,
            reranker=name,
            engine=src.source,
            model=src.model,
            file=src.file if isinstance(src, GgufSource) else None,
            method=src.method,
            max_items=src.max_items,
            max_chars=src.max_chars,
            cases=self.cases,
            shards=src.shards,
            concurrency=src.concurrency,
            forward_batch=src.forward_batch if isinstance(src, LayaSource) else 1,
            gpu=None if isinstance(src, ApiSource) else src.gpu,
            api=(
                modal_app.ApiEndpoint(url=src.url, secret=src.secret, key_env=src.key_env)
                if isinstance(src, ApiSource)
                else None
            ),
        )


def load_config(path: Path) -> Experiment:
    return Experiment.model_validate(yaml.safe_load(path.read_text()))


def new_id() -> str:
    return f"{time.strftime(STAMP, time.gmtime())}_{petname.generate(2)}"


class Paths:
    def __init__(self, id: str, root: Path = EXPERIMENTS_DIR) -> None:
        self.id = id
        self.dir = root / id

    @property
    def config(self) -> Path:
        return self.dir / f"config_{self.id}.yaml"

    @property
    def calls(self) -> Path:
        return self.dir / f"calls_{self.id}.jsonl"

    def raw(self, reranker: str) -> Path:
        gz = self.dir / f"raw_{reranker}_{self.id}.json.gz"
        plain = gz.with_suffix("")  # experiments imported before answers were gzipped
        return plain if plain.exists() and not gz.exists() else gz

    @property
    def report_base(self) -> Path:
        return self.dir / f"kept_mass_{self.id}"

    @property
    def costs(self) -> Path:
        return self.dir / f"costs_{self.id}.json"

    @property
    def latency(self) -> Path:
        return self.dir / f"latency_{self.id}.json"


def run(config: Path, wait: bool) -> str:
    exp = load_config(config)
    p = Paths(new_id())
    p.dir.mkdir(parents=True)
    shutil.copy(config, p.config)
    print(f"experiment {p.id}: {p.dir}")
    runs = [
        exp.scoring_run(p.id, name, src, config.as_posix())
        for name, src in exp.rerankers.items()
        if isinstance(src, ModelSource)
    ]
    if runs:
        modal_app.deploy()
        for r in runs:
            modal_app.spawn(r, p.calls)
    if wait:
        finish(p.id)
    return p.id


def status(id: str) -> None:
    for rec, state in modal_app.states(Paths(id).calls):
        print(f"{rec.spawned_at}  {rec.job.desc:60s} {state}")


def resume(id: str) -> None:
    """Re-spawn every shard whose latest call failed; finished batches on the Volume are kept."""
    latest = {rec.job.desc: (rec.job, state) for rec, state in modal_app.states(Paths(id).calls)}
    failed = [job for job, state in latest.values() if state.startswith("FAILED")]
    if failed:
        modal_app.deploy()
    for job in failed:
        modal_app.spawn_shard(job, Paths(id).calls)


# Modal list prices, $/GPU-second (modal.com/pricing, 2026-09); RAM $0.00000222/GiB/s added per shard.
GPU_USD_PER_S = {
    "L4": 0.000222,
    "L40S": 0.000542,
    "A100-80GB": 0.000694,
    "RTX-PRO-6000": 0.000842,
    "H100": 0.001097,
    "H200": 0.001261,
}
RAM_USD_PER_GIB_S = 0.00000222


def _usd_per_s(gpu: str, model: str) -> float:
    gib = (modal_app.MEMORY_MB_FOR.get(model) or 0) / 1024
    return GPU_USD_PER_S[gpu] + gib * RAM_USD_PER_GIB_S


def costs(id: str) -> str:
    """Per-reranker GPU-seconds and $ from the shards' own timers: load, warm, in-function.

    Every spawned shard counts; the latest call per shard is the one billed for its answers, earlier
    failed calls are counted as unmeasured. Writes costs_<id>.md and costs_<id>.json."""
    p = Paths(id)
    exp = load_config(p.config)
    n_cases = len(load_cases()[: exp.cases])
    shards: dict[str, list] = {}
    models: dict[str, str] = {}
    for rec, state in modal_app.states(p.calls):
        shards.setdefault(rec.job.run.reranker, []).append(parse_summary(state))
        models[rec.job.run.reranker] = rec.job.run.model
    rows: dict[str, CostRow] = {}
    for name, summaries in shards.items():
        gpus = {s.gpu for s in summaries if s is not None}
        if len(gpus) > 1:
            raise ValueError(f"{name}: shards on several GPU types {sorted(gpus)}")
        # a hosted API bills tokens, not seconds: no GPU price here
        rate = _usd_per_s(next(iter(gpus)), models[name]) if gpus - {modal_app.API} else 0.0
        rows[name] = shard_costs(summaries, rate)
    per_1k = 1000 / n_cases
    out = {
        "cases": n_cases,
        "rerankers": {
            n: {**r.model_dump(), "warm_usd_per_1k": r.warm_usd * per_1k} for n, r in rows.items()
        },
        "shards": {
            n: [None if s is None else s.model_dump() for s in ss] for n, ss in shards.items()
        },
    }
    p.costs.write_text(json.dumps(out, indent=1) + "\n")
    lines = [
        f"# costs {id}",
        "",
        "Kev/Laya: the shard's own timers (metrics.py) x Modal list price (GPU + reserved host RAM).",
        "warm = loaded server to last answer; in-function = container function start to return.",
        "prod / Jev / API rows are token costs from their own billing (not re-billed here).",
        "",
        f"USD = whole run ({n_cases} queries once); per 1k queries = linear extrapolation.",
        "",
        "| ranker | gpu | shards | unmeasured | load s | warm GPU s | in-function s | warm USD | warm USD / 1k |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in exp.rerankers:
        r = rows.get(name)
        if r is None:
            lines.append(f"| {name} | - | 0 | - | - | - | - | (API / frozen) | |")
            continue
        lines.append(
            f"| {name} | {r.gpu} | {r.shards} | {r.unmeasured} | {r.load_s:.0f} | {r.warm_gpu_s:.0f}"
            f" | {r.in_function_s:.0f} | {r.warm_usd:.2f} | {r.warm_usd * per_1k:.2f} |"
        )
    md = p.dir / f"costs_{id}.md"
    md.write_text("\n".join(lines) + "\n")
    return str(md)


def latency(id: str) -> str:
    """Per-query wall clock of this experiment's Modal-scored rerankers: first request sent to
    last answer, with the shard's other queries in flight. Writes latency_<id>.json and .csv.
    prod / Jev latency is in data/timing_summary_prod_jev.csv."""
    p = Paths(id)
    exp = load_config(p.config)
    rows = [
        "method,queries,mean_s_per_query,median_s_per_query,p95_s_per_query,min_s_per_query,"
        "max_s_per_query,mean_calls_per_query"
    ]
    out: dict[str, dict] = {}
    for name, src in exp.rerankers.items():
        if not isinstance(src, ModelSource) or not p.raw(name).exists():
            continue
        records = read_raw(p.raw(name))
        per = query_wall_s(records)
        v = sorted(per.values())
        n = len(v)
        stats = {
            "mean_s": sum(v) / n,
            "p50_s": v[n // 2],
            "p95_s": v[min(n - 1, int(0.95 * n))],
            "min_s": v[0],
            "max_s": v[-1],
        }
        out[name] = {"kind": "wall_clock", "queries": n, **stats, "per_case_s": per}
        rows.append(
            f"{name},{n},{stats['mean_s']:.2f},{stats['p50_s']:.2f},{stats['p95_s']:.2f},"
            f"{stats['min_s']:.2f},{stats['max_s']:.2f},{len(records) / n:.1f}"
        )
    p.latency.write_text(json.dumps({"rerankers": out}, indent=1) + "\n")
    csv = p.dir / f"latency_{id}.csv"
    csv.write_text("\n".join(rows) + "\n")
    return str(csv)


def _wait(calls: Path) -> None:
    pending = [rec for rec, state in modal_app.states(calls) if state == "running"]
    with tqdm(total=len(pending), desc="shards finished", unit="shard") as bar:
        while pending:
            time.sleep(15)
            still = [rec for rec, state in modal_app.states(calls) if state == "running"]
            bar.update(len(pending) - len(still))
            pending = still


def _slice(records: list[RawRecord], cases: int | None) -> list[RawRecord]:
    """Only the experiment's cases (answer files may cover more, e.g. Jev's 200-case run)."""
    keep = {c.id for c in load_cases()[:cases]}
    return [r for r in records if r.case_id in keep]


def finish(id: str) -> None:
    """Pull every reranker's answers into the experiment dir and write its table + graph."""
    p = Paths(id)
    exp = load_config(p.config)
    unfinished: set[str] = set()
    if p.calls.exists():
        _wait(p.calls)
        latest = {rec.job.desc: (rec.job.run.reranker, s) for rec, s in modal_app.states(p.calls)}
        unfinished = {name for name, state in latest.values() if state.startswith("FAILED")}
    rerankers: list[Reranker] = []
    for name, src in exp.rerankers.items():
        if isinstance(src, ProductionSource):
            rerankers.append(ProductionReranker(name))
            continue
        if name in unfinished:  # partial answers would score a different set of cases
            print(f"skipped {name}: a shard failed or was cancelled (`resume` to finish it)")
            continue
        if isinstance(src, AnswersSource):
            run_name = src.path.name.split(".")[0]
            records = _slice(read_raw(REPO_DIR / src.path), exp.cases)
        else:
            scoring = exp.scoring_run(id, name, src)
            run_name = scoring.name
            records = modal_app.pull(scoring)
        dest = write_raw(p.raw(name), run_name, records)
        cases = {r.case_id for r in records}
        print(f"wrote {dest}: {len(records)} requests over {len(cases)} cases")
        rerankers.append(ScoredReranker(name, METHODS[src.method], dest))
    if not any(isinstance(s, ProductionSource) and n == PROD for n, s in exp.rerankers.items()):
        print(f"no reranker named {PROD!r} with source production: no win-rates")
    per_case, n_cases = evaluate(rerankers, exp.k.ks, exp.cases)
    means, wins = tables(per_case, exp.k.ks)
    print(write_report(means, wins, n_cases, p.report_base))
    if p.calls.exists():
        print(Path(costs(id)).read_text())
        print(Path(latency(id)).read_text())


class RerankerEvidence(BaseModel):
    model_config = {"frozen": True}

    name: str
    requests: int
    cases: int


class Evidence(BaseModel):
    """What proves an experiment really ran: every Modal call finished, every scored reranker
    answered every case, and the kept-mass table was written. `problems` is empty when it did."""

    model_config = {"frozen": True}

    experiment: str
    calls: dict[str, str]  # Modal call id -> "finished" | "running" | "FAILED: ..."
    rerankers: list[RerankerEvidence]
    kept_mass: bool
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


RUNNING = "running"
FAILED = "FAILED"
FINISHED = "finished"


def call_states(calls: Path) -> dict[str, str]:
    """Each spawned Modal call reduced to finished / running / FAILED: <why>."""
    out: dict[str, str] = {}
    for rec, state in modal_app.states(calls):
        if state == RUNNING or state.startswith(FAILED):
            out[rec.call_id] = state
        else:
            out[rec.call_id] = FINISHED
    return out


def evidence(id: str, calls: dict[str, str]) -> Evidence:
    """Offline part of `verify`: judge the experiment directory given its Modal call states."""
    p = Paths(id)
    exp = load_config(p.config)
    expected = exp.cases or len(load_cases())
    problems = [
        f"modal call {call_id}: {state}" for call_id, state in calls.items() if state != FINISHED
    ]
    rerankers: list[RerankerEvidence] = []
    for name, src in exp.rerankers.items():
        if isinstance(src, ProductionSource):
            continue
        if not p.raw(name).exists():
            problems.append(f"{name}: no raw answers")
            continue
        records = read_raw(p.raw(name))
        answered = len({r.case_id for r in records})
        rerankers.append(RerankerEvidence(name=name, requests=len(records), cases=answered))
        if answered != expected:
            problems.append(f"{name}: {answered} of {expected} cases answered")
    kept_mass = p.report_base.with_suffix(".json").exists()
    if not kept_mass:
        problems.append(f"no {p.report_base.with_suffix('.json').name}")
    return Evidence(
        experiment=id, calls=calls, rerankers=rerankers, kept_mass=kept_mass, problems=problems
    )


def verify(id: str) -> Evidence:
    p = Paths(id)
    return evidence(id, call_states(p.calls) if p.calls.exists() else {})


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("config", type=Path, help="configs/<name>.yaml")
    r.add_argument("--wait", action="store_true", help="then finish once every shard is done")
    sub.add_parser("status").add_argument("id")
    sub.add_parser("finish").add_argument("id")
    sub.add_parser("resume").add_argument("id")
    sub.add_parser("costs").add_argument("id")
    sub.add_parser("latency").add_argument("id")
    sub.add_parser(
        "verify", help="exit 1 unless every call finished and every case is answered"
    ).add_argument("id")
    sub.add_parser("logs")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.config, a.wait)
    elif a.cmd == "status":
        status(a.id)
    elif a.cmd == "finish":
        finish(a.id)
    elif a.cmd == "resume":
        resume(a.id)
    elif a.cmd == "costs":
        print(Path(costs(a.id)).read_text())
    elif a.cmd == "latency":
        print(Path(latency(a.id)).read_text())
    elif a.cmd == "verify":
        found = verify(a.id)
        print(found.model_dump_json(indent=1))
        if not found.ok:
            sys.exit(1)
    else:
        modal_app.logs()


if __name__ == "__main__":
    main()

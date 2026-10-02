"""Score decision models on Modal, in-process, over the 75 cases.

experiment.py deploys this app under APP_NAME from the working tree and ``.spawn()``s a scoring
function into it once per shard (case i -> shard i % shards, one GPU each), so the run outlives
the laptop; each shard appends its answers to the runs Volume (resumable per request) and ``pull``
merges the shards back into RawRecords. Every producer answers the same `SystemOneRequest` and is
validated into the same `SystemOneResponse`:

* Kev (``score_cases``): Hugging Face weights through ``kev.Checkpoint``, asked through
  ``kev.serve.Server.answer`` (its model thread batches whatever is queued).
* Laya (``score_cases_laya``): the ModernBERT decision encoder, many one-child requests per
  forward pass (laya_batch).
* Clef (``score_cases_clef``): Cloudflare's Qwen3.5 backbone + joint schema head, asked through
  the release's own ``joint_schema_model.systemone`` (one request per forward pass).
* MATILDA (``score_cases_matilda``): Maincode's Qwen3.5 backbone + decision readout, asked through
  the release's own runtime (``maincode_jev_serve.decide``), one child per request.
* AutoTrust (``score_cases_autotrust``): Qwen3.8-27B + LoRA + 24-slot decision head, the release's
  bare prompt per question (jev_tracker.autotrust), one child per request.
* JevAny (``score_cases_jevany``): a pointer LoRA on a Qwen3.5 base through the JevAny release's
  ``JevModel``, which answers the request body in-process.
* RSI-Jev (``score_cases_rsi_jev``): a fine-tuned Qwen3.5 tower + option scorer, loaded by the
  release's ``load_release`` and scored with its ``rsijev.evaluate.predict`` (jev_tracker.rsi_jev
  maps the request body to its typed questions, as its server's wire.py does).
* MiniCPM5-Jev (``score_cases_minicpm_jev``): a LoRA + letter readout on MiniCPM5-2B through the
  release's ``MiniCPMSystemOne``, which converts the request body itself (jev_tracker.minicpm_jev).
* StartLux-Decision (``score_cases_startlux``): a Qwen3.5 decoder read out at the option letters,
  through the release's own ``startlux_decision.StartLuxDecision.decide``, which answers the
  request body in-process (eager padded passes; the server's CUDA graphs are off).
* Von (``score_cases_von``): wfzyx/von, a 395M encoder + option-marker head behind the ``von-sdk``
  ``VonEngine``, which takes the request's raw question dicts (jev_tracker.von adds the ``type``
  back to its answers).
* Bekko (``score_cases_bekko``): hotchpotch's bekko-system-one v0 encoders (17M/68M/400M), asked
  through the release's ``BekkoSentenceTransformer.predict``; jev_tracker.bekko turns each
  request into one input object (shared state, one judgment decision per question).
* API (``score_cases_api``): any hosted model that answers the System One request at a URL; a CPU
  container posts the bodies unchanged with a bearer key from a Modal Secret. The only scorer
  that leaves the container.

Modal credentials come from MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment.
"""

import functools
import gzip
import io
import os
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import modal
from pydantic import BaseModel
from tqdm import tqdm

from jev_tracker.contract import DataPoint, load_cases
from jev_tracker.methods import METHODS, Item, Method, batches, record, request
from jev_tracker.metrics import ShardSummary
from jev_tracker.rerankers import append_jsonl_gz, read_jsonl_gz
from jev_tracker.systemone import RawRecord, SystemOneResponse

APP_NAME = "jev-tracker"
RUNS_VOLUME_NAME = "jev-tracker-runs"
HF_CACHE_VOLUME_NAME = "rox-research--hf-cache"  # Kev / Laya weights already cached there
REPO_DIR = Path(__file__).resolve().parents[1]
REMOTE_DIR = "/root/jev-tracker"
RUNS_ROOT = Path("/runs")
REMOTE_RAW_DIR = RUNS_ROOT / "raw"
HF_CACHE_DIR = "/root/.cache/huggingface"
PYTHON_VERSION = "3.12"
HOUR = 60 * 60
COMMIT_EVERY_S = 60
PROGRESS_DICT_NAME = "jev-tracker-progress"
PROGRESS_LABEL = "jev-tracker-progress"  # web endpoint: https://<workspace>--<label>.modal.run
PROGRESS_EVERY_S = 5  # the bars' tqdm mininterval, so the site sees what the terminal sees
PROGRESS_STALE_S = 24 * HOUR  # a bar not touched for this long is no longer served
FASTAPI_PKG = "fastapi[standard]>=0.115"  # Modal's web endpoints are FastAPI handlers

# Runtime deps of jev_tracker inside the containers (the repo is mounted, not installed).
RUNTIME_DEPS = ("pydantic>=2.13.4", "tqdm>=4.67", "httpx>=0.28.1", "pyyaml>=6.0.3")
# github.com/jaredpalmer/kev, main on 2026-09-25; releases: kev-0.8b / kev-4b / kev-9b / kev-27b.
KEV_REF = "a4d8705fc9ca07a93bcc581d81904cf3cb3ae9e1"
KEV_PKG = f"kev[serve] @ git+https://github.com/jaredpalmer/kev.git@{KEV_REF}"
# Installed after kev: torch pins triton 3.4, which fla refuses on Hopper (fla#640).
KEV_KERNELS = ("flash-linear-attention==0.5.2", "triton>=3.7.1")
# github.com/NandhaKishorM/laya: 0.3.5 is the newest release at least 7 days old on 2026-09-29.
LAYA_PKG = "laya==0.3.5"
LAYA_MODEL = "convaiinnovations/laya"
# The three official checkpoints, all read from LAYA_MODEL's pinned bundle revision:
# model id -> its subfolder there (None = the repo root).
LAYA_SUBFOLDER = {
    LAYA_MODEL: None,  # ModernBERT-large, 421M, 512 tokens, English
    "convaiinnovations/laya-multilingual": "multilingual",  # mmBERT-base, 322M, 1,024 tokens
    "convaiinnovations/laya-typed-decisions": "typed-decisions",  # laya fine-tuned, 1,024 tokens
}
LAYA_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
LAYA_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]
# huggingface.co/Cloudflare/clef{,-flash}: the release card is "tested with torch 2.11 and
# transformers 5.10.2"; torchvision + pillow back its image/video processor.
CLEF_PKGS = (
    "torch==2.11.0",
    "torchvision==0.26.0",
    "transformers==5.10.2",
    "accelerate>=1.10",
    "safetensors>=0.6",
    "pillow>=11",
)
# model id -> pinned Hub revision (2026-10-02); weights + joint head + joint_schema_model.py.
CLEF_REVISION = {
    "Cloudflare/clef": "2f3de3dd85f379784083b0814d997ab627200f0c",  # 27.4B, 55 GB bf16
    "Cloudflare/clef-flash": "17f0b0ad64efb65d273590632833508766b2aae6",  # 9.4B, 19 GB bf16
}
# huggingface.co/Maincode/matilda-jev-v1 (2026-10-02): USAGE.txt "tested ... Transformers 5.17.0,
# PyTorch 2.14.0" plus its requirements-runtime.txt; runtime/ holds maincode_jev_serve.
MATILDA_MODEL = "Maincode/matilda-jev-v1"
MATILDA_REVISION = "c87f57504300588ff4270ff5ac17c38d1c7996b9"  # 26.1B, 49 GiB bf16
MATILDA_PKGS = (
    "torch==2.14.0",
    "torchvision==0.29.0",
    "transformers==5.17.0",
    "flash-linear-attention==0.5.2",
    "safetensors==0.8.0",
    "pillow==12.3.0",
    "numpy==2.5.3",
)
MATILDA_BATCH = 64  # rows per forward pass, the runtime's default (MJ_BATCH_SIZE)
# huggingface.co/autotrust/JEV-27B (2026-10-01): README "torch 2.13 + cu130, transformers 5.16,
# peft 0.21, flash-linear-attention 0.5.2"; transformers 5.17.0 as MATILDA (config says 5.16.1).
AUTOTRUST_MODEL = "autotrust/JEV-27B"
AUTOTRUST_REVISION = "962701f3e5437ef5d7cceedae98741477007d367"  # 27B, 54 GB bf16
AUTOTRUST_PKGS = (
    "torch==2.14.0",
    "transformers==5.17.0",
    "peft==0.21.0",
    "accelerate>=1.10",
    "safetensors>=0.6",
    "flash-linear-attention==0.5.2",
)
# Weights, tokenizer, adapter/ (the HF LoRA), head.safetensors, judge_config.json, calibration.json.
AUTOTRUST_FILES = [
    "*.json",
    "model-*.safetensors",
    "head.safetensors",
    "adapter/*",
    "tokenizer*",
    "chat_template.jinja",
]
# github.com/SimpleJev/JevAny, main on 2026-10-02 (0.3.0): the code release the model cards require.
JEVANY_REF = "f8701e76bb61f7cc60e5a40125ed6a8361c9ab87"
JEVANY_PKG = f"jevany[local,fast] @ git+https://github.com/SimpleJev/JevAny.git@{JEVANY_REF}"
# The base's Qwen3.5 processor config makes transformers load an image processor (torchvision, Pillow).
JEVANY_PKGS = (JEVANY_PKG, "torch==2.14.0", "torchvision==0.29.0", "pillow==12.3.0")
# huggingface.co/shgao/rsi-jev-v3.0-qwen3.5-2b (2026-10-02): code/ holds load_release + rsijev; the
# base Qwen/Qwen3.5-2B-Base is read from the Hub by load_release (main: b1485b2f on 2026-10-02).
# Kernel stack as the release card (fla 0.5.2); torchvision + pillow back the base's processor.
RSI_JEV_REVISION = {
    "shgao/rsi-jev-v3.0-qwen3.5-2b": "c778b68dd5c20e67ac7ca8d8ef1f9a1258a687a5",
}
RSI_JEV_PKGS = (
    "torch==2.14.0",
    "torchvision==0.29.0",
    "transformers==5.17.0",
    "flash-linear-attention==0.5.2",
    "safetensors==0.8.0",
    "pillow==12.3.0",
)
RSI_JEV_BATCH = 4  # questions per forward pass (each repeats the whole state)
# huggingface.co/ytbai/MiniCPM5-2B-Jev (2026-10-02): model.py + adapter + head.pt; its loader reads
# the base openbmb/MiniCPM5-2B (a Llama, main: f9740005 on 2026-10-02) from the Hub.
MINICPM_JEV_REVISION = {
    "ytbai/MiniCPM5-2B-Jev": "27afc6f17ceb7eb95b6442047ec54e43fbed5ed1",
}
# Its requirements.txt floors; peft as the adapter's peft_version.
MINICPM_JEV_PKGS = (
    "torch==2.14.0",
    "transformers==5.17.0",
    "peft==0.21.0",
    "safetensors==0.8.0",
    "numpy==2.5.3",
)
# huggingface.co/startlux-models/StartLux-Decision-9B (2026-10-02): weights + decision_config.json +
# startlux_decision/ (Apache-2.0 runtime). requirements.txt: "tested with torch 2.11, transformers
# 5.8.1, flash-linear-attention 0.5.2, causal-conv1d 1.7.0"; causal-conv1d 1.7.0 ships wheels up to
# torch 2.10 (cu12), so torch 2.10.0 cu128 + that wheel; transformers 5.8.1 still exports the
# qwen3_5 `is_fast_path_available` its loader checks.
STARTLUX_REVISION = {
    "startlux-models/StartLux-Decision-9B": "2974b71c4bc6ba9766f3fbca804fd0ce54db223c",
}
STARTLUX_TORCH = ("torch==2.10.0",)
STARTLUX_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
CAUSAL_CONV1D_WHEEL = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/"
    "causal_conv1d-1.7.0%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)
STARTLUX_PKGS = (
    "transformers==5.8.1",
    "accelerate>=1.10",  # its loader passes device_map
    "huggingface_hub>=0.34",
    "safetensors>=0.4",
    "flash-linear-attention==0.5.2",
    CAUSAL_CONV1D_WHEEL,
)
JEVANY_REVISION = {
    "SimpleJev/JevAny-Qwen3.5-4B-LoRA": "1c7aa9bab14ac347aeb917c0bcd757838a8a78ce",
}
# huggingface.co/hotchpotch/bekko-system-one-v0-* (2026-10-02): inference_v0.py + the Ettin-based
# weights; the card's dependency pins (torch <2.11, transformers/sentence-transformers exact).
BEKKO_REVISION = {
    "hotchpotch/bekko-system-one-v0-17m": "b886a1f9b91f4e8368d7830080d52d955e2c8dfa",
    "hotchpotch/bekko-system-one-v0-68m": "ab7685f23e5edbc1acb12ced4f2c4e12591efa69",
    "hotchpotch/bekko-system-one-v0-400m": "1960df5602bd93cc8d336fdebd9fb68a30926e13",
}
BEKKO_PKGS = (
    "torch==2.10.0",
    "transformers==5.17.0",
    "sentence-transformers==6.1.0",
    "safetensors>=0.7",
)
# huggingface.co/wfzyx/von (2026-10-02): encoder + option_marker.pt read by von-sdk's VonEngine
# via hf_hub_download without a revision; the pinned snapshot is warmed and HF_HUB_OFFLINE=1
# keeps the engine on it. von-sdk is the release's package (Von 1.3).
VON_REPO = "wfzyx/von"
VON_REVISION = "498ceba33390b32cfefaab6422ec380318ba9b99"
VON_PKG = "von-sdk==1.3.7"
GPU_FOR = {
    "jaredpalmer/kev-0.8b": "L4",
    "Glax147/kev-0.8b-ba-lora": "L4",  # a Kev-0.8B post-train: as Kev-0.8B
    "jaredpalmer/kev-4b": "L40S",
    "jaredpalmer/kev-9b": "H100",
    "jaredpalmer/kev-27b": "H200",  # 55 GB bf16 weights; 25-item states OOM an 80 GB H100
    **{m: "L4" for m in LAYA_SUBFOLDER},
    "Cloudflare/clef": "H200",  # as Kev-27B: the budget line
    "Cloudflare/clef-flash": "H100",
    MATILDA_MODEL: "H200",  # 26.1B, as Kev-27B: the budget line
    AUTOTRUST_MODEL: "H200",  # 27B, as Kev-27B: the budget line
    "SimpleJev/JevAny-Qwen3.5-4B-LoRA": "L40S",  # as Kev-4B
    **{m: "L40S" for m in RSI_JEV_REVISION},  # 2B, fp32 tower (the release's evaluation precision)
    **{m: "L4" for m in MINICPM_JEV_REVISION},  # 2B bf16, as Kev-0.8B
    "Glax147/kev-4b-ba-lora": "L40S",  # a Kev-4B post-train: as Kev-4B
    **{m: "H100" for m in STARTLUX_REVISION},  # 9B bf16, as Kev-9B
    VON_REPO: "L4",  # 395M encoder, as Laya
    **{m: "L4" for m in BEKKO_REVISION},  # 17M-400M encoders, as Laya
}
DEFAULT_KEV_GPU = "H100"  # a Kev checkpoint not in GPU_FOR
# Kev-27B stages its bf16 weights through host memory while loading (upstream: --memory-mb 131072).
MEMORY_MB_FOR = {
    "jaredpalmer/kev-27b": 131_072,
    "Cloudflare/clef": 131_072,
    MATILDA_MODEL: 131_072,
    AUTOTRUST_MODEL: 131_072,
}
API = "API"  # gpu of a hosted-API scorer: no GPU is attached or billed
API_TRANSIENT = {429, 500, 502, 503, 504}
API_RETRIES = 16  # waits 2, 4, ... 60 s, about 12 min in all
API_BACKOFF_S, API_MAX_WAIT_S = 2.0, 60.0


class ApiEndpoint(BaseModel):
    """A hosted System One endpoint: where to post, and which Modal Secret holds the bearer key."""

    model_config = {"frozen": True}

    url: str
    secret: str | None = None  # Modal Secret name; None = no Authorization header
    key_env: str = "API_KEY"  # the variable inside that Secret


class ScoringRun(BaseModel):
    """One model's scoring run: a reranker of one experiment."""

    model_config = {"frozen": True}

    experiment: str  # experiment id; answers live under raw/<experiment>/<reranker>/ on the Volume
    reranker: str
    engine: Literal[
        "kev",
        "laya",
        "clef",
        "matilda",
        "autotrust",
        "jevany",
        "rsi_jev",
        "minicpm_jev",
        "startlux",
        "von",
        "bekko",
        "api",
    ]
    model: str
    method: str
    max_items: int | None  # children per System One request; None = the whole case
    max_chars: int | None  # characters of child text per request; None = no limit
    cases: int | None = None  # first N cases only (smoke); None = all 75
    shards: int = 1  # containers (GPUs) the cases are dealt over
    # Requests kept in flight per container (client threads). Kev's Server queues them and its
    # model thread runs everything queued as one batch, so >1 = fewer, fuller forward passes.
    concurrency: int = 1
    forward_batch: int = 1  # Laya only: one-child requests per GPU forward pass (laya_batch)
    gpu: str | None = None  # Modal GPU type; None = GPU_FOR[model]
    api: ApiEndpoint | None = None  # api only
    config: str | None = None  # configs/<name>.yaml this run came from: the site's queue item key

    @property
    def repo(self) -> str:
        """The Hub repo id without any `@revision` pin (kev's Checkpoint accepts `repo@rev`)."""
        return self.model.partition("@")[0]

    @property
    def gpu_type(self) -> str:
        if self.engine == "api":
            return API
        return self.gpu or GPU_FOR.get(self.repo, DEFAULT_KEV_GPU)

    @property
    def name(self) -> str:
        short = self.model.split("/")[-1]
        i = "all" if self.max_items is None else self.max_items
        c = "all" if self.max_chars is None else self.max_chars
        return f"{short}_{self.method}_i{i}_c{c}"

    @property
    def remote_dir(self) -> Path:
        return REMOTE_RAW_DIR / self.experiment / self.reranker

    def shard_path(self, shard: int) -> Path:
        return self.remote_dir / f"shard{shard:02d}of{self.shards:02d}.jsonl.gz"

    def shard_cases(self, shard: int) -> list[DataPoint]:
        return load_cases()[: self.cases][shard :: self.shards]


class ShardJob(BaseModel):
    model_config = {"frozen": True}

    run: ScoringRun
    shard: int

    @property
    def desc(self) -> str:
        return f"{self.run.reranker} ({self.run.name}) shard {self.shard}/{self.run.shards}"


class CallRecord(BaseModel):
    model_config = {"frozen": True}

    job: ShardJob
    call_id: str
    spawned_at: str


app = modal.App(APP_NAME)


def _base() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("git")
        .pip_install(*RUNTIME_DEPS)
        .env({"HF_HOME": HF_CACHE_DIR, "TOKENIZERS_PARALLELISM": "false"})
        .workdir(REMOTE_DIR)
    )


def _mount(image: modal.Image) -> modal.Image:
    """The package and the frozen benchmark; experiments and the site stay on the laptop."""
    return image.add_local_dir(
        str(REPO_DIR / "jev_tracker"), f"{REMOTE_DIR}/jev_tracker", ignore=["**/__pycache__"]
    ).add_local_dir(str(REPO_DIR / "data" / "benchmark"), f"{REMOTE_DIR}/data/benchmark")


kev_image = _mount(
    _base()
    .pip_install(KEV_PKG)
    .pip_install(*KEV_KERNELS)
    .env({"TRITON_CACHE_DIR": f"{HF_CACHE_DIR}/triton-cache"})
)
laya_image = _mount(_base().pip_install(LAYA_PKG))  # brings torch + transformers
clef_image = _mount(_base().pip_install(*CLEF_PKGS))
matilda_image = _mount(_base().pip_install(*MATILDA_PKGS))
autotrust_image = _mount(_base().pip_install(*AUTOTRUST_PKGS))
jevany_image = _mount(_base().pip_install(*JEVANY_PKGS))
rsi_jev_image = _mount(_base().pip_install(*RSI_JEV_PKGS))
minicpm_jev_image = _mount(_base().pip_install(*MINICPM_JEV_PKGS))
startlux_image = _mount(
    _base().pip_install(*STARTLUX_TORCH, index_url=STARTLUX_TORCH_INDEX).pip_install(*STARTLUX_PKGS)
)
von_image = _mount(_base().pip_install(VON_PKG))  # brings torch + transformers
bekko_image = _mount(_base().pip_install(*BEKKO_PKGS))
api_image = _mount(_base())
progress_image = _mount(_base().pip_install(FASTAPI_PKG))

runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
VOLUMES = {str(RUNS_ROOT): runs_volume, HF_CACHE_DIR: hf_cache}
progress_dict = modal.Dict.from_name(PROGRESS_DICT_NAME, create_if_missing=True)


class ShardProgress(BaseModel):
    """One shard's tqdm bar as data, so the site can draw it on the queue item `config`."""

    model_config = {"frozen": True}

    config: str | None
    experiment: str
    reranker: str
    shard: int
    shards: int
    done: int  # requests answered so far, the resumed ones included
    total: int  # requests in this shard
    resumed: int  # answered by an earlier call of this shard, so not part of this call's rate
    started_at: float  # epoch seconds this call began answering
    updated_at: float

    @property
    def key(self) -> str:
        return f"{self.experiment}/{self.reranker}/{self.shard:02d}"

    @property
    def finished(self) -> bool:
        return self.done >= self.total


def shard_progress(
    job: ShardJob, done: int, total: int, resumed: int, started_at: float, now: float
) -> ShardProgress:
    return ShardProgress(
        config=job.run.config,
        experiment=job.run.experiment,
        reranker=job.run.reranker,
        shard=job.shard,
        shards=job.run.shards,
        done=done,
        total=total,
        resumed=resumed,
        started_at=started_at,
        updated_at=now,
    )


class ProgressBar:
    """Publishes a shard's bar to `store` at most every PROGRESS_EVERY_S seconds, and at the end."""

    def __init__(
        self,
        job: ShardJob,
        resumed: int,
        total: int,
        store: modal.Dict = progress_dict,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.job, self.resumed, self.total, self.store, self.clock = (
            job,
            resumed,
            total,
            store,
            clock,
        )
        self.started_at = clock()
        self.published_at = float("-inf")
        self.update(resumed)

    def update(self, done: int) -> None:
        now = self.clock()
        if now - self.published_at < PROGRESS_EVERY_S and done < self.total:
            return
        bar = shard_progress(self.job, done, self.total, self.resumed, self.started_at, now)
        self.store.put(bar.key, bar.model_dump())
        self.published_at = now


def _answer_shard(
    job: ShardJob,
    done: set[tuple[str, int]],
    answer: Callable[[DataPoint, int, list[Item]], None],
) -> int:
    """Every not-yet-answered batch of the shard's cases, committed to the Volume as it goes.

    `run.concurrency` client threads each drive one request at a time through `answer`, so that
    many requests are queued at the model together (Kev's Server batches whatever is queued)."""
    run = job.run
    todo = [
        (case, bi, batch)
        for case in run.shard_cases(job.shard)
        for bi, batch in enumerate(batches(case.input, run.max_items, run.max_chars))
        if (case.id, bi) not in done
    ]
    last_commit = time.time()
    n = 0
    bar = ProgressBar(job, resumed=len(done), total=len(done) + len(todo))
    with ThreadPoolExecutor(max_workers=run.concurrency, thread_name_prefix="client") as pool:
        results = pool.map(lambda t: answer(*t), todo)
        for _ in tqdm(results, total=len(todo), desc=job.desc, unit="batch", mininterval=5):
            n += 1
            bar.update(len(done) + n)
            if time.time() - last_commit > COMMIT_EVERY_S:
                runs_volume.commit()
                last_commit = time.time()
    runs_volume.commit()
    return n


def _answer_forward_shard(
    job: ShardJob,
    done: set[tuple[str, int]],
    predict: Callable[[list[dict]], list[dict]],
) -> tuple[int, int]:
    """(requests, forward passes): the shard's unanswered requests in passes of
    `run.forward_batch` bodies per `predict` call, `run.concurrency` calls in flight.

    For engines whose predict already takes a list (Bekko's input objects, RSI-Jev's
    cases): one call answers a whole chunk, the true batch; the threads keep more
    passes in flight on top of it."""
    run = job.run
    method: Method = METHODS[run.method]
    out = run.shard_path(job.shard)
    todo = [
        (case, bi, batch)
        for case in run.shard_cases(job.shard)
        for bi, batch in enumerate(batches(case.input, run.max_items, run.max_chars))
        if (case.id, bi) not in done
    ]
    chunks = [todo[lo : lo + run.forward_batch] for lo in range(0, len(todo), run.forward_batch)]
    last_commit = time.time()
    n = passes = 0
    bar = ProgressBar(job, resumed=len(done), total=len(done) + len(todo))

    def answer(chunk: list[tuple[DataPoint, int, list[Item]]]):
        bodies = [
            request(method, case.input.query, batch, run.model).body() for case, _bi, batch in chunk
        ]
        t = time.time()
        resps = predict(bodies)
        return chunk, resps, t, time.time() - t

    pbar = tqdm(total=len(todo), desc=job.desc, unit="batch", mininterval=5)
    with ThreadPoolExecutor(max_workers=run.concurrency, thread_name_prefix="client") as pool:
        for chunk, resps, t, dt in pool.map(answer, chunks):
            for (case, bi, batch), resp in zip(chunk, resps, strict=True):
                resp_obj = SystemOneResponse.model_validate(resp)
                rec = record(
                    method,
                    case.id,
                    bi,
                    batch,
                    resp_obj,
                    dt,
                    t,
                    resp_obj.usage.input_tokens or None,
                )
                append_jsonl_gz(out, rec)
            n += len(chunk)
            passes += 1
            pbar.update(len(chunk))
            bar.update(len(done) + n)
            if time.time() - last_commit > COMMIT_EVERY_S:
                runs_volume.commit()
                last_commit = time.time()
    pbar.close()
    runs_volume.commit()
    return n, passes


@app.function(
    image=kev_image,
    gpu="L40S",
    timeout=4 * HOUR,
    single_use_containers=True,  # one shard per container: a reused one still holds the last model
    volumes=VOLUMES,
)
def score_cases(job_json: str) -> str:
    """Answer every batch of this shard's cases with Kev; resumable ((case, batch) done = skipped)."""
    import torch
    from fastapi import HTTPException
    from kev.api import SystemOneRequest, to_record
    from kev.checkpoint import Checkpoint, LoadOptions, fused_available
    from kev.model import SERVE_MAX_BRANCH, SERVE_MAX_STATE
    from kev.serve import Server, prepare

    job = ShardJob.model_validate_json(job_json)
    run = job.run
    method = METHODS[run.method]
    out = run.shard_path(job.shard)
    done = {(r.case_id, r.batch) for r in read_jsonl_gz(out)}

    t0 = time.time()
    ck = Checkpoint(run.model)
    tok, model = ck.load(
        "cuda", LoadOptions(dtype=torch.bfloat16, cuda_graphs=True, fused=fused_available())
    )
    server = Server(ck, tok, model, "cuda")  # kev.serve's own construction, minus uvicorn
    loaded = time.time()
    resident_gb = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()  # peak below = scoring only, not the bf16 load
    print(f"loaded {run.model} in {loaded - t0:.0f}s: {ck.path}, {resident_gb:.1f} GB", flush=True)
    write = threading.Lock()  # append_jsonl_gz from the client threads, one at a time

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> None:
        req = request(method, case.input.query, batch, "kev-latest")
        kev_req = SystemOneRequest.model_validate(req.body())  # our contract -> kev.api's
        t = time.time()
        try:
            body = server.answer(kev_req)
        except HTTPException as e:
            if e.status_code != 422 or len(batch) == 1:  # 422 = over Kev's state token limit
                raise
            half = len(batch) // 2
            answer(case, bi, batch[:half])
            answer(case, bi, batch[half:])
            return
        latency_s = time.time() - t  # incl. queue wait
        resp = SystemOneResponse.model_validate(body)  # kev.serve's answer -> our contract
        enc = model.encode(
            tok,
            to_record(prepare(kev_req))[0],
            max_state=SERVE_MAX_STATE,
            max_branch=SERVE_MAX_BRANCH,
        )
        rec = record(method, case.id, bi, batch, resp, latency_s, t, enc["seg"].count(0))
        with write:
            append_jsonl_gz(out, rec)

    n = _answer_shard(job, done, answer)
    scored = time.time()
    server.close()
    return ShardSummary(
        reranker=run.reranker,
        shard=job.shard,
        gpu=run.gpu_type.rstrip("!"),
        load_s=loaded - t0,
        warm_s=scored - loaded,
        total_s=time.time() - t0,
        requests=n,
        resumed=len(done),
        device=torch.cuda.get_device_name(),
        resident_gb=resident_gb,
        peak_gb=torch.cuda.max_memory_allocated() / 1e9,
        forward_passes=server.batches,
        graphs=model.graphs.stats() if model.graphs is not None else None,
    ).model_dump_json()


@app.function(
    image=laya_image,
    gpu=GPU_FOR[LAYA_MODEL],
    timeout=4 * HOUR,
    single_use_containers=True,
    volumes=VOLUMES,
)
def score_cases_laya(job_json: str) -> str:
    """Answer every batch of this shard's cases with Laya in-process; resumable like score_cases.

    The /v1/systemone request bodies go in unchanged; `run.forward_batch` of a case's requests
    share one forward pass (laya_batch), so a query's latency is its passes back to back."""
    from huggingface_hub import snapshot_download
    from laya import Agent

    from jev_tracker import laya_batch  # imports laya, which only the Laya image has

    job = ShardJob.model_validate_json(job_json)
    run = job.run
    out = run.shard_path(job.shard)
    done = {(r.case_id, r.batch) for r in read_jsonl_gz(out)}

    t0 = time.time()
    sub = LAYA_SUBFOLDER[run.model]
    files = [f"{sub}/{f}" if sub else f for f in LAYA_FILES]
    ckpt = snapshot_download(LAYA_MODEL, revision=LAYA_REVISION, allow_patterns=files)
    agent = Agent(ckpt, device="cuda", subfolder=sub)
    if agent.device.type != "cuda":  # laya falls back to CPU silently
        raise RuntimeError(f"laya loaded {run.model} on {agent.device}, not cuda")
    load_s = time.time() - t0
    print(f"loaded {run.model}@{LAYA_REVISION[:12]} on {agent.device} {agent.dtype}", flush=True)

    n, passes = _answer_laya_shard(job, done, functools.partial(laya_batch.predict_batch, agent))
    return ShardSummary(
        reranker=run.reranker,
        shard=job.shard,
        gpu=run.gpu_type.rstrip("!"),
        load_s=load_s,
        warm_s=time.time() - t0 - load_s,
        total_s=time.time() - t0,
        requests=n,
        resumed=len(done),
        forward_passes=passes,
    ).model_dump_json()


def _answer_laya_shard(
    job: ShardJob,
    done: set[tuple[str, int]],
    predict: Callable[[list[dict]], list[dict]],
) -> tuple[int, int]:
    """(requests, forward passes): each case's unanswered requests, shortest child first, in
    passes of `run.forward_batch`; every request records its pass's start and duration."""
    run = job.run
    method: Method = METHODS[run.method]
    out = run.shard_path(job.shard)
    last_commit = time.time()
    n = passes = 0
    cases = run.shard_cases(job.shard)
    total = sum(len(batches(case.input, run.max_items, run.max_chars)) for case in cases)
    bar = ProgressBar(job, resumed=len(done), total=total)
    for case in tqdm(cases, desc=job.desc, unit="case", mininterval=5):
        todo = sorted(
            (
                (bi, batch)
                for bi, batch in enumerate(batches(case.input, run.max_items, run.max_chars))
                if (case.id, bi) not in done
            ),
            key=lambda t: sum(len(item.text) for item in t[1]),
        )
        for lo in range(0, len(todo), run.forward_batch):
            chunk = todo[lo : lo + run.forward_batch]
            bodies = [
                request(method, case.input.query, batch, run.model).body() for _, batch in chunk
            ]
            t = time.time()
            resps = predict(bodies)
            dt = time.time() - t
            for (bi, batch), resp in zip(chunk, resps, strict=True):
                rec = record(
                    method, case.id, bi, batch, SystemOneResponse.model_validate(resp), dt, t
                )
                append_jsonl_gz(out, rec)
            n += len(chunk)
            passes += 1
            bar.update(len(done) + n)
        if time.time() - last_commit > COMMIT_EVERY_S:
            runs_volume.commit()
            last_commit = time.time()
    runs_volume.commit()
    return n, passes


@app.function(
    image=clef_image,
    gpu="H100",
    timeout=4 * HOUR,
    single_use_containers=True,
    volumes=VOLUMES,
)
def score_cases_clef(job_json: str) -> str:
    """Answer every batch of this shard's cases with Clef in-process; resumable like score_cases.

    The release's `joint_schema_model.systemone` takes the /v1/systemone request body unchanged
    and runs one forward pass per request, so `run.concurrency` is 1 (ClefSource)."""
    import sys

    import torch
    from huggingface_hub import snapshot_download

    job = ShardJob.model_validate_json(job_json)
    run = job.run
    method = METHODS[run.method]
    out = run.shard_path(job.shard)
    done = {(r.case_id, r.batch) for r in read_jsonl_gz(out)}

    t0 = time.time()
    path = snapshot_download(run.repo, revision=CLEF_REVISION[run.repo])
    sys.path.insert(0, path)
    import joint_schema_model  # the release's own loader and /v1/systemone answerer

    model, processor = joint_schema_model.load_release_model(path, device="cuda")
    loaded = time.time()
    resident_gb = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    print(f"loaded {run.repo}@{CLEF_REVISION[run.repo][:12]} in {loaded - t0:.0f}s", flush=True)

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> None:
        body = request(method, case.input.query, batch, run.model).body()
        t = time.time()
        resp = SystemOneResponse.model_validate(
            joint_schema_model.systemone(model, processor, body)
        )
        latency_s = time.time() - t
        rec = record(method, case.id, bi, batch, resp, latency_s, t, resp.usage.input_tokens)
        append_jsonl_gz(out, rec)

    n = _answer_shard(job, done, answer)
    scored = time.time()
    return ShardSummary(
        reranker=run.reranker,
        shard=job.shard,
        gpu=run.gpu_type.rstrip("!"),
        load_s=loaded - t0,
        warm_s=scored - loaded,
        total_s=time.time() - t0,
        requests=n,
        resumed=len(done),
        device=torch.cuda.get_device_name(),
        resident_gb=resident_gb,
        peak_gb=torch.cuda.max_memory_allocated() / 1e9,
        forward_passes=n,
    ).model_dump_json()


def _summary(
    job: ShardJob, t0: float, loaded: float, n: int, resumed: int, resident_gb: float
) -> str:
    """ShardSummary of a one-request-per-forward-pass GPU scorer (timers since container start)."""
    import torch

    run = job.run
    return ShardSummary(
        reranker=run.reranker,
        shard=job.shard,
        gpu=run.gpu_type.rstrip("!"),
        load_s=loaded - t0,
        warm_s=time.time() - loaded,
        total_s=time.time() - t0,
        requests=n,
        resumed=resumed,
        device=torch.cuda.get_device_name(),
        resident_gb=resident_gb,
        peak_gb=torch.cuda.max_memory_allocated() / 1e9,
        forward_passes=n,
    ).model_dump_json()


@app.function(
    image=matilda_image,
    gpu="H200",
    timeout=4 * HOUR,
    single_use_containers=True,
    volumes=VOLUMES,
)
def score_cases_matilda(job_json: str) -> str:
    """Answer every batch of this shard's cases with MATILDA in-process; resumable like score_cases.

    The release's runtime/ (`maincode_jev_serve`) loads the checkpoint and scores the request's
    state + questions (`decide`, then `answer`), exactly as its /v1/systemone server does."""
    import sys

    import torch
    from huggingface_hub import snapshot_download

    job = ShardJob.model_validate_json(job_json)
    run = job.run
    method = METHODS[run.method]
    out = run.shard_path(job.shard)
    done = {(r.case_id, r.batch) for r in read_jsonl_gz(out)}

    t0 = time.time()
    path = snapshot_download(run.repo, revision=MATILDA_REVISION)
    sys.path.insert(0, f"{path}/runtime")
    from maincode_jev_serve.config import gpu_limits
    from maincode_jev_serve.decide import decide
    from maincode_jev_serve.model import DecisionModel
    from maincode_jev_serve.model import answer as decision_answer

    model = DecisionModel(checkpoint=path, device="cuda")
    model.eval()
    max_tokens, token_budget = gpu_limits()
    loaded = time.time()
    resident_gb = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    print(f"loaded {run.repo}@{MATILDA_REVISION[:12]} in {loaded - t0:.0f}s", flush=True)

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> None:
        body = request(method, case.input.query, batch, run.model).body()
        t = time.time()
        distributions, input_tokens = decide(
            model,
            body["state"],
            body["questions"],
            temperature=model.temperature,
            max_tokens=max_tokens,
            token_budget=token_budget,
            batch_size=MATILDA_BATCH,
        )
        resp = SystemOneResponse.model_validate(
            {
                "model": run.model,
                "answers": {
                    key: decision_answer(body["questions"][key], values)
                    for key, values in distributions.items()
                },
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            }
        )
        latency_s = time.time() - t
        rec = record(method, case.id, bi, batch, resp, latency_s, t, input_tokens)
        append_jsonl_gz(out, rec)

    n = _answer_shard(job, done, answer)
    return _summary(job, t0, loaded, n, len(done), resident_gb)


@app.function(
    image=autotrust_image,
    gpu="H200",
    timeout=4 * HOUR,
    single_use_containers=True,
    volumes=VOLUMES,
)
def score_cases_autotrust(job_json: str) -> str:
    """Answer every batch of this shard's cases with AutoTrust JEV-27B; resumable like score_cases.

    The README's transformers path: LoRA merged into the backbone, the last token's hidden state
    through the 24-slot head, one forward pass per question (jev_tracker.autotrust)."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from jev_tracker import autotrust

    job = ShardJob.model_validate_json(job_json)
    run = job.run
    method = METHODS[run.method]
    out = run.shard_path(job.shard)
    done = {(r.case_id, r.batch) for r in read_jsonl_gz(out)}

    t0 = time.time()
    path = Path(
        snapshot_download(run.repo, revision=AUTOTRUST_REVISION, allow_patterns=AUTOTRUST_FILES)
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, str(path), subfolder="adapter").merge_and_unload()
    model.eval()
    weights = load_file(str(path / "head.safetensors"))
    head_weight = weights["proj.weight"].cuda().float()
    head_bias = weights["proj.bias"].cuda().float()
    head = autotrust.Head.load(path)
    loaded = time.time()
    resident_gb = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    print(f"loaded {run.repo}@{AUTOTRUST_REVISION[:12]} in {loaded - t0:.0f}s", flush=True)

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> None:
        body = request(method, case.input.query, batch, run.model).body()
        asked = autotrust.prompts(body)
        t = time.time()
        logits: dict[str, list[float]] = {}
        tokens = 0
        for prompt in asked:
            ids = tokenizer(prompt.text, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model.model(**ids).last_hidden_state[0, -1].float()
            logits[prompt.key] = (head_weight @ hidden + head_bias).tolist()
            tokens += ids["input_ids"].shape[1]
        resp = SystemOneResponse.model_validate(
            autotrust.response(run.model, asked, logits, head, tokens)
        )
        latency_s = time.time() - t
        rec = record(method, case.id, bi, batch, resp, latency_s, t, tokens)
        append_jsonl_gz(out, rec)

    n = _answer_shard(job, done, answer)
    return _summary(job, t0, loaded, n, len(done), resident_gb)


@app.function(
    image=jevany_image,
    gpu="L40S",
    timeout=4 * HOUR,
    single_use_containers=True,
    volumes=VOLUMES,
)
def score_cases_jevany(job_json: str) -> str:
    """Answer every batch of this shard's cases with a JevAny checkpoint; resumable like score_cases.

    `jevany.JevModel` takes the /v1/systemone request body unchanged; a request over its packed
    token limit (ValueError "exceeds") is split in two, as score_cases does on Kev's 422."""
    import torch
    from jevany import JevModel

    job = ShardJob.model_validate_json(job_json)
    run = job.run
    method = METHODS[run.method]
    out = run.shard_path(job.shard)
    done = {(r.case_id, r.batch) for r in read_jsonl_gz(out)}

    t0 = time.time()
    revision = JEVANY_REVISION[run.repo]
    jev = JevModel.from_pretrained(f"{run.repo}@{revision}", device="cuda", dtype="bf16")
    served = jev.runtime.model_id
    loaded = time.time()
    resident_gb = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    print(f"loaded {run.repo}@{revision[:12]} as {served} in {loaded - t0:.0f}s", flush=True)

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> None:
        body = request(method, case.input.query, batch, served).body()
        t = time.time()
        try:
            resp = SystemOneResponse.model_validate(jev(body))
        except ValueError as e:
            if "exceeds" not in str(e) or len(batch) == 1:
                raise
            half = len(batch) // 2
            answer(case, bi, batch[:half])
            answer(case, bi, batch[half:])
            return
        latency_s = time.time() - t
        rec = record(method, case.id, bi, batch, resp, latency_s, t, resp.usage.input_tokens)
        append_jsonl_gz(out, rec)

    n = _answer_shard(job, done, answer)
    return _summary(job, t0, loaded, n, len(done), resident_gb)


@app.function(
    image=rsi_jev_image,
    gpu="L40S",
    timeout=4 * HOUR,
    single_use_containers=True,
    volumes=VOLUMES,
)
def score_cases_rsi_jev(job_json: str) -> str:
    """Answer every batch of this shard's cases with an RSI-Jev release; resumable like score_cases.

    The release's `load_release` rebuilds the scored model (fp32 tower, fp32 scorer, its fitted
    calibration) and `rsijev.evaluate.predict` scores each question as one encoded row; the
    request body becomes its typed questions through jev_tracker.rsi_jev. As its server, the state
    is not cut (max_length = rsi_jev.MAX_INPUT_TOKENS)."""
    import dataclasses
    import sys

    import torch
    from huggingface_hub import snapshot_download

    from jev_tracker import rsi_jev

    job = ShardJob.model_validate_json(job_json)
    run = job.run
    out = run.shard_path(job.shard)
    done = {(r.case_id, r.batch) for r in read_jsonl_gz(out)}

    t0 = time.time()
    revision = RSI_JEV_REVISION[run.repo]
    path = snapshot_download(run.repo, revision=revision)
    sys.path.insert(0, f"{path}/code")
    from load_release import load_release
    from rsijev.contract import Case, Question
    from rsijev.encode import encode_question
    from rsijev.evaluate import predict as rsi_predict

    model, tokenizer, encode_config, meta = load_release(path, device="cuda")
    encode_config = dataclasses.replace(encode_config, max_length=rsi_jev.MAX_INPUT_TOKENS)
    max_options = meta["spec"]["max_options"]
    loaded = time.time()
    resident_gb = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    print(
        f"loaded {run.repo}@{revision[:12]} (calibration {meta['calibration']}) in "
        f"{loaded - t0:.0f}s",
        flush=True,
    )

    def predict(bodies: list[dict]) -> list[dict]:
        requests_ = []
        for body in bodies:
            asked = {q.key: q for q in rsi_jev.questions(body)}
            state = rsi_jev.state_text(body)
            typed = tuple(Question(**q.model_dump()) for q in asked.values())
            uniform = {q.key: tuple(1 / len(q.options) for _ in q.options) for q in typed}
            rsi_case = Case(
                case_id="x", source="jev-tracker", state=state, questions=typed, gold=uniform
            )
            input_tokens = sum(
                len(encode_question(tokenizer, state, q, encode_config)["input_ids"]) for q in typed
            )
            requests_.append((body, asked, rsi_case, input_tokens))
        predictions = rsi_predict(
            model,
            tokenizer,
            [rsi_case for _, _, rsi_case, _ in requests_],
            encode_config,
            max_options=max_options,
            device="cuda",
            batch_size=RSI_JEV_BATCH,
        )
        index_of = {id(rsi_case): i for i, (_, _, rsi_case, _) in enumerate(requests_)}
        grouped: list[list] = [[] for _ in requests_]
        for rsi_case, q, prediction in predictions:
            grouped[index_of[id(rsi_case)]].append((q, prediction))
        return [
            {
                "model": run.model,
                "answers": {
                    q.key: rsi_jev.answer(asked[q.key], list(prediction.probs))
                    for q, prediction in grouped[i]
                },
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            }
            for i, (body, asked, rsi_case, input_tokens) in enumerate(requests_)
        ]

    n, _forward_passes = _answer_forward_shard(job, done, predict)
    return _summary(job, t0, loaded, n, len(done), resident_gb)


@app.function(
    image=minicpm_jev_image,
    gpu="L4",
    timeout=4 * HOUR,
    single_use_containers=True,
    volumes=VOLUMES,
)
def score_cases_minicpm_jev(job_json: str) -> str:
    """Answer every batch of this shard's cases with MiniCPM5-2B-Jev; resumable like score_cases.

    The release's model.py loads the checkpoint (`MiniCPMSystemOne.load_checkpoint`, bf16),
    converts the request body (`to_internal_record`, date facts on) and scores every question in
    one pass at its serving limits, exactly as its /v1/systemone server does (PriDe off)."""
    import sys

    import torch
    from huggingface_hub import snapshot_download

    from jev_tracker import minicpm_jev

    job = ShardJob.model_validate_json(job_json)
    run = job.run
    method = METHODS[run.method]
    out = run.shard_path(job.shard)
    done = {(r.case_id, r.batch) for r in read_jsonl_gz(out)}

    t0 = time.time()
    revision = MINICPM_JEV_REVISION[run.repo]
    path = snapshot_download(run.repo, revision=revision)
    sys.path.insert(0, path)
    from model import (
        SERVE_MAX_BRANCH,
        SERVE_MAX_STATE,
        MiniCPMSystemOne,
        encode_record,
        to_internal_record,
    )

    model, tokenizer = MiniCPMSystemOne.load_checkpoint(path, device="cuda", dtype=torch.bfloat16)
    loaded = time.time()
    resident_gb = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    print(f"loaded {run.repo}@{revision[:12]} in {loaded - t0:.0f}s", flush=True)

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> None:
        body = request(method, case.input.query, batch, run.model).body()
        internal = to_internal_record(body, add_date_facts=True)
        encoded = encode_record(
            tokenizer,
            internal,
            max_state=SERVE_MAX_STATE,
            max_branch=SERVE_MAX_BRANCH,
            strict=False,
        )
        t = time.time()
        outputs = model.logits_and_probs(encoded)
        input_tokens = len(encoded["ids"])
        resp = SystemOneResponse.model_validate(
            minicpm_jev.response(
                run.model,
                internal["questions"],
                [probs.float().tolist() for _, probs in outputs],
                input_tokens,
            )
        )
        latency_s = time.time() - t
        rec = record(method, case.id, bi, batch, resp, latency_s, t, input_tokens)
        append_jsonl_gz(out, rec)

    n = _answer_shard(job, done, answer)
    return _summary(job, t0, loaded, n, len(done), resident_gb)


@app.function(
    image=startlux_image,
    gpu="H100",
    timeout=4 * HOUR,
    single_use_containers=True,
    volumes=VOLUMES,
)
def score_cases_startlux(job_json: str) -> str:
    """Answer every batch of this shard's cases with StartLux-Decision; resumable like score_cases.

    The release's `StartLuxDecision` (bf16, fast kernels required) renders one prompt per question
    and reads the option letters' logits, all questions of a request in one padded pass; its
    `decide` answers the request body in the System One format, as its /v1/systemone server does.
    CUDA graphs are off: the eager path is the one the server falls back to."""
    import sys

    import torch
    from huggingface_hub import snapshot_download

    job = ShardJob.model_validate_json(job_json)
    run = job.run
    method = METHODS[run.method]
    out = run.shard_path(job.shard)
    done = {(r.case_id, r.batch) for r in read_jsonl_gz(out)}

    t0 = time.time()
    revision = STARTLUX_REVISION[run.repo]
    path = snapshot_download(run.repo, revision=revision)
    sys.path.insert(0, path)
    from startlux_decision import StartLuxDecision

    model = StartLuxDecision(path, device="cuda", graphs=False)
    loaded = time.time()
    resident_gb = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    print(
        f"loaded {run.repo}@{revision[:12]} in {loaded - t0:.0f}s "
        f"(fast kernels: {model.fast_kernels})",
        flush=True,
    )

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> None:
        body = request(method, case.input.query, batch, run.model).body()
        t = time.time()
        answers, usage = model.decide(body["state"], body["questions"])
        resp = SystemOneResponse.model_validate(
            {"model": run.model, "answers": answers, "usage": usage}
        )
        latency_s = time.time() - t
        rec = record(method, case.id, bi, batch, resp, latency_s, t, usage["input_tokens"])
        append_jsonl_gz(out, rec)

    n = _answer_shard(job, done, answer)
    return _summary(job, t0, loaded, n, len(done), resident_gb)


@app.function(
    image=von_image,
    gpu="L4",
    timeout=4 * HOUR,
    single_use_containers=True,
    volumes=VOLUMES,
)
def score_cases_von(job_json: str) -> str:
    """Answer every batch of this shard's cases with Von in-process; resumable like score_cases.

    The SDK's `VonEngine.evaluate` takes the request's state and raw question dicts unchanged
    (as its /v1/systemone server does); jev_tracker.von puts the answer `type` back. The pinned
    snapshot is warmed and HF_HUB_OFFLINE=1 so the engine's unpinned hf_hub_download reads it."""
    import torch
    from huggingface_hub import snapshot_download
    from von.engine import VonEngine

    from jev_tracker import von

    job = ShardJob.model_validate_json(job_json)
    run = job.run
    method = METHODS[run.method]
    out = run.shard_path(job.shard)
    done = {(r.case_id, r.batch) for r in read_jsonl_gz(out)}

    t0 = time.time()
    snapshot_download(run.repo, revision=VON_REVISION)
    os.environ["HF_HUB_OFFLINE"] = "1"
    engine = VonEngine.get_instance(device="cuda")
    loaded = time.time()
    resident_gb = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    print(f"loaded {run.repo}@{VON_REVISION[:12]} in {loaded - t0:.0f}s", flush=True)

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> None:
        body = request(method, case.input.query, batch, run.model).body()
        t = time.time()
        result = engine.evaluate(state=body["state"], questions=body["questions"], model=run.model)
        resp = SystemOneResponse.model_validate(von.response(run.model, body["questions"], result))
        latency_s = time.time() - t
        rec = record(method, case.id, bi, batch, resp, latency_s, t, resp.usage.input_tokens)
        append_jsonl_gz(out, rec)

    n = _answer_shard(job, done, answer)
    return _summary(job, t0, loaded, n, len(done), resident_gb)


@app.function(
    image=bekko_image,
    gpu="L4",
    timeout=4 * HOUR,
    single_use_containers=True,
    volumes=VOLUMES,
)
def score_cases_bekko(job_json: str) -> str:
    """Answer every batch of this shard's cases with a Bekko checkpoint; resumable like score_cases.

    The release's `inference_v0.BekkoSentenceTransformer.predict` runs one input object per
    request (jev_tracker.bekko maps it): the state shared, one judgment decision per question."""
    import torch
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from jev_tracker import bekko

    job = ShardJob.model_validate_json(job_json)
    run = job.run
    out = run.shard_path(job.shard)
    done = {(r.case_id, r.batch) for r in read_jsonl_gz(out)}

    t0 = time.time()
    revision = BEKKO_REVISION[run.repo]
    model_cls = get_class_from_dynamic_module(
        "inference_v0.BekkoSentenceTransformer", run.repo, revision=revision
    )
    model = model_cls(run.repo, trust_remote_code=True, device="cuda", revision=revision)
    loaded = time.time()
    resident_gb = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    print(
        f"loaded {run.repo}@{revision[:12]} in {loaded - t0:.0f}s "
        f"(attn {model[0].attn_implementation})",
        flush=True,
    )

    def predict(bodies: list[dict]) -> list[dict]:
        inputs = [bekko.input_object(body) for body in bodies]
        results = model.predict(inputs, show_progress_bar=False)
        return [
            {
                "model": run.model,
                "answers": {
                    key: bekko.answer(body["questions"][key], result[key])
                    for key in body["questions"]
                },
                "usage": {"input_tokens": 0, "output_tokens": 0},  # predict reports no tokens
            }
            for body, result in zip(bodies, results, strict=True)
        ]

    n, _forward_passes = _answer_forward_shard(job, done, predict)
    return _summary(job, t0, loaded, n, len(done), resident_gb)


@app.function(image=api_image, timeout=4 * HOUR, single_use_containers=True, volumes=VOLUMES)
def score_cases_api(job_json: str) -> str:
    """Post every batch of this shard's cases to a hosted System One endpoint; resumable.

    `run.concurrency` requests in flight. A batch the endpoint rejects as over its input limit
    (400/422) is split in two, as score_cases does on Kev's 422. A transient error is retried
    with exponential backoff; a record's latency is its successful attempt."""
    import httpx

    job = ShardJob.model_validate_json(job_json)
    run = job.run
    if run.api is None:
        raise ValueError(f"{job.desc}: api source without an endpoint")
    method = METHODS[run.method]
    out = run.shard_path(job.shard)
    done = {(r.case_id, r.batch) for r in read_jsonl_gz(out)}
    t0 = time.time()
    headers = {}
    if run.api.secret is not None:
        headers["Authorization"] = f"Bearer {os.environ[run.api.key_env]}"
    client = httpx.Client(headers=headers, timeout=httpx.Timeout(600.0, connect=10.0))
    write = threading.Lock()

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> None:
        body = request(method, case.input.query, batch, run.model).body()
        for attempt in range(API_RETRIES + 1):
            t = time.time()
            r = client.post(run.api.url, json=body)
            if r.status_code not in API_TRANSIENT or attempt == API_RETRIES:
                break
            print(f"{case.id} batch {bi}: HTTP {r.status_code}, retry {attempt + 1}", flush=True)
            time.sleep(min(API_BACKOFF_S * 2**attempt, API_MAX_WAIT_S))
        if r.status_code in (400, 422) and len(batch) > 1:
            half = len(batch) // 2
            answer(case, bi, batch[:half])
            answer(case, bi, batch[half:])
            return
        if r.status_code != 200:
            raise RuntimeError(f"{case.id} batch {bi}: HTTP {r.status_code}: {r.text[:300]}")
        rec = record(
            method,
            case.id,
            bi,
            batch,
            SystemOneResponse.model_validate(r.json()),
            time.time() - t,
            t,
        )
        with write:
            append_jsonl_gz(out, rec)

    loaded = time.time()
    n = _answer_shard(job, done, answer)
    scored = time.time()
    client.close()
    return ShardSummary(
        reranker=run.reranker,
        shard=job.shard,
        gpu=API,
        load_s=loaded - t0,
        warm_s=scored - loaded,
        total_s=time.time() - t0,
        requests=n,
        resumed=len(done),
    ).model_dump_json()


@app.function(image=progress_image)
@modal.fastapi_endpoint(label=PROGRESS_LABEL)
def progress() -> dict:
    """Every shard bar touched in the last PROGRESS_STALE_S; the site polls it (CORS on, nothing secret)."""
    now = time.time()
    shards = [bar for bar in progress_dict.values() if now - bar["updated_at"] < PROGRESS_STALE_S]
    return {"at": now, "shards": sorted(shards, key=lambda bar: bar["updated_at"], reverse=True)}


def deploy() -> None:
    """Publish this working tree as APP_NAME; every later spawn runs this code."""
    with modal.enable_output():
        app.deploy(name=APP_NAME)
    print(f"progress feed: {progress.get_web_url()}", flush=True)


def spawn(run: ScoringRun, calls_file: Path) -> None:
    """One call per shard, appended to the experiment's calls file. Call deploy() first."""
    for shard in range(run.shards):
        spawn_shard(ShardJob(run=run, shard=shard), calls_file)


def spawn_shard(job: ShardJob, calls_file: Path) -> None:
    """Spawn (or re-spawn: done batches on the Volume are skipped) one shard and record its call."""
    score = {
        "kev": score_cases,
        "laya": score_cases_laya,
        "clef": score_cases_clef,
        "matilda": score_cases_matilda,
        "autotrust": score_cases_autotrust,
        "jevany": score_cases_jevany,
        "rsi_jev": score_cases_rsi_jev,
        "minicpm_jev": score_cases_minicpm_jev,
        "startlux": score_cases_startlux,
        "von": score_cases_von,
        "bekko": score_cases_bekko,
        "api": score_cases_api,
    }
    run = job.run
    options: dict = {"memory": MEMORY_MB_FOR.get(run.repo)}
    if run.engine == "api":
        if run.api is not None and run.api.secret is not None:
            options["secrets"] = [modal.Secret.from_name(run.api.secret)]
    else:
        options["gpu"] = run.gpu_type
    fn = score[run.engine].with_options(**options)
    call = fn.spawn(job.model_dump_json())
    rec = CallRecord(
        job=job,
        call_id=call.object_id,
        spawned_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    calls_file.parent.mkdir(parents=True, exist_ok=True)
    with calls_file.open("a") as f:
        f.write(rec.model_dump_json() + "\n")
    print(f"spawned {call.object_id}: {job.desc} ({len(run.shard_cases(job.shard))} cases)")


def calls(calls_file: Path) -> list[CallRecord]:
    if not calls_file.exists():
        return []
    return [
        CallRecord.model_validate_json(line) for line in calls_file.read_text().splitlines() if line
    ]


def states(calls_file: Path) -> list[tuple[CallRecord, str]]:
    """Each spawned call with its return value, or "running"."""
    out = []
    for rec in calls(calls_file):
        call = modal.FunctionCall.from_id(rec.call_id)
        try:
            state = call.get(timeout=0)
        except TimeoutError:
            state = "running"
        except Exception as e:  # noqa: BLE001 - the shard's own exception, surfaced as its state
            state = f"FAILED: {type(e).__name__}: {str(e).splitlines()[-1][:200]}"
        out.append((rec, state))
    return out


def logs() -> None:
    """Follow the app's container output (download + scoring tqdm bars); Ctrl-C to stop."""
    subprocess.run(["modal", "app", "logs", APP_NAME, "--follow", "--timestamps"])


def pull(run: ScoringRun) -> list[RawRecord]:
    """Merge a run's shards from the Volume, sorted by (case, batch)."""
    vol = modal.Volume.from_name(RUNS_VOLUME_NAME)
    remote = run.remote_dir.relative_to(RUNS_ROOT)
    entries = sorted(vol.listdir(str(remote)), key=lambda e: e.path)
    if len(entries) != run.shards:
        raise FileNotFoundError(f"{remote}: {len(entries)} of {run.shards} shard files")
    records: list[RawRecord] = []
    for entry in tqdm(entries, desc=f"pull {run.reranker}", unit="shard"):
        with io.BytesIO(b"".join(vol.read_file(entry.path))) as buf, gzip.open(buf, "rt") as f:
            records += [RawRecord.model_validate_json(line) for line in f if line.strip()]
    records.sort(key=lambda r: (r.case_id, r.batch))
    return records

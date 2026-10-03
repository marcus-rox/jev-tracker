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
import hashlib
import io
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Literal

import httpx
import modal
from pydantic import BaseModel
from tqdm import tqdm

from jev_tracker.contract import DataPoint, load_cases
from jev_tracker.methods import METHODS, Item, Method, batches, items, record, request
from jev_tracker.metrics import ShardSummary
from jev_tracker.rerankers import append_jsonl_gz, read_jsonl_gz
from jev_tracker.systemone import RawRecord, SystemOneResponse

# JEV_TRACKER_APP names a separate deployment: two working trees deploying the same app replace
# each other's functions and kill each other's running shards.
APP_NAME = os.environ.get("JEV_TRACKER_APP", "jev-tracker")
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
PROGRESS_LABEL = f"{APP_NAME}-progress"  # web endpoint: https://<workspace>--<label>.modal.run
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
API_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
# github.com/ggml-org/llama.cpp master of 2026-10-02 18:26 UTC. /v1/systemone (#29818, merged 09:56)
# is in no release yet (b11351 = the 08:08 commit, its qwen35 loader has no decision head), so
# llama-server is built from source, statically, for the GPUs GPU_USD_PER_S prices (L40S, H100).
LLAMA_CPP_COMMIT = "bed0a856606ee4a24a164066f73d2379447033f5"
LLAMA_CPP_SRC = "/opt/llama.cpp-src"
LLAMA_CPP_DIR = "/opt/llama.cpp"
LLAMA_CPP_CUDA_ARCHS = "89;90"  # sm_89 = L4 / L40S, sm_90 = H100
LLAMA_CPP_BUILD_CPUS = 32  # the CUDA kernels are ~150 nvcc jobs; the default builder takes >1 h
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"  # nvcc to build, libcudart / libcublas to run
LLAMA_CPP_APT = ("git", "cmake", "build-essential", "ca-certificates")
HF_HUB_PKG = "huggingface_hub>=0.30"
LLAMA_URL = "http://127.0.0.1:8080"
LLAMA_LOG = Path("/tmp/llama-server.log")
LLAMA_START_TIMEOUT_S = 10 * 60  # includes reading a 5 GB file from the Volume
LLAMA_POLL_S = 2.0
LLAMA_CTX_PER_SLOT = (
    8192  # tokens per in-flight request; --ctx-size is shared over --parallel slots
)
DEFAULT_GGUF_GPU = "L40S"
# ghcr.io/ollaya-dev/ollaya:cuda12 of 2026-10-02 (Ollaya 0.9.0: /usr/bin/ollaya + ONNX Runtime CUDA 12
# libraries), pinned by digest. cuda12 rather than cuda: the CUDA 13 pack needs an R580+ driver.
OLLAYA_IMAGE = "ghcr.io/ollaya-dev/ollaya@sha256:e613df044a10c4deb2b29a6996c96d83ef2bc897670369234f321eec99a04c47"
OLLAYA_BIN = "/usr/bin/ollaya"
OLLAYA_URL = "http://127.0.0.1:11435"
OLLAYA_MODELS_DIR = f"{HF_CACHE_DIR}/ollaya/models"  # the model store, on the HF-cache Volume
OLLAYA_LOG = Path("/tmp/ollaya.log")
OLLAYA_START_TIMEOUT_S = 60
OLLAYA_LOAD_TIMEOUT = "30m"  # bf16 Kev-9B is ~20 GB read from the Volume into ORT; default 5m
OLLAYA_PULL_TIMEOUT = httpx.Timeout(HOUR, connect=10.0)
DEFAULT_OLLAYA_GPU = "H100"


class ApiEndpoint(BaseModel):
    """A hosted System One endpoint: where to post, and which Modal Secret holds the bearer key."""

    model_config = {"frozen": True}

    url: str
    secret: str | None = None  # Modal Secret name; None = no Authorization header
    key_env: str = "API_KEY"  # the variable inside that Secret


class SweepPoint(BaseModel):
    """One batch-size probe: `requests` System One requests of `children` items each."""

    model_config = {"frozen": True}

    requests: int
    children: int


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
        "gguf",
        "ollaya",
    ]
    model: str
    file: str | None = None  # gguf only: the .gguf file inside the `model` repo (Q4_K_M, Q8_0, ...)
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
    # Latency runs: `shards` is the GPU pool; dispatch_cases deals one case's requests over it.
    fanout: bool = False
    sweep: list[SweepPoint] | None = None  # batch-size probes instead of the case pass

    @property
    def queue_name(self) -> str:
        """The Modal Queue the dispatcher and this run's workers share."""
        return "fanout-" + hashlib.sha1((self.experiment + self.reranker).encode()).hexdigest()[:16]

    @property
    def repo(self) -> str:
        """The Hub repo id without any `@revision` pin (kev's Checkpoint accepts `repo@rev`)."""
        return self.model.partition("@")[0]

    @property
    def gpu_type(self) -> str:
        if self.engine == "api":
            return API
        if self.engine == "gguf":
            return self.gpu or DEFAULT_GGUF_GPU
        if self.engine == "ollaya":
            return self.gpu or DEFAULT_OLLAYA_GPU
        return self.gpu or GPU_FOR.get(self.repo, DEFAULT_KEV_GPU)

    @property
    def name(self) -> str:
        short = self.model.split("/")[-1]
        if self.file is not None:
            short = Path(self.file).stem
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
    role: Literal["worker", "dispatch"] = "worker"

    @property
    def desc(self) -> str:
        if self.role == "dispatch":
            return f"{self.run.reranker} ({self.run.name}) dispatch"
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


def _build_llama_server() -> None:
    """Image build step: a static CUDA llama-server at LLAMA_CPP_COMMIT, on a many-core builder."""
    src, build = LLAMA_CPP_SRC, f"{LLAMA_CPP_SRC}/build"
    for cmd in (
        ["git", "clone", "https://github.com/ggml-org/llama.cpp", src],
        ["git", "-C", src, "checkout", LLAMA_CPP_COMMIT],
        [
            "cmake",
            "-S",
            src,
            "-B",
            build,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={LLAMA_CPP_CUDA_ARCHS}",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
        ],
        ["cmake", "--build", build, "--target", "llama-server", "-j", str(LLAMA_CPP_BUILD_CPUS)],
    ):
        subprocess.run(cmd, check=True)
    Path(LLAMA_CPP_DIR).mkdir(parents=True, exist_ok=True)
    shutil.copy(f"{build}/bin/llama-server", f"{LLAMA_CPP_DIR}/llama-server")
    shutil.rmtree(src)


gguf_image = _mount(
    modal.Image.from_registry(CUDA_IMAGE, add_python=PYTHON_VERSION)
    .apt_install(*LLAMA_CPP_APT)
    .pip_install(*RUNTIME_DEPS, HF_HUB_PKG)
    .run_function(_build_llama_server, cpu=LLAMA_CPP_BUILD_CPUS, memory=64 * 1024, timeout=HOUR)
    .env({"HF_HOME": HF_CACHE_DIR})
    .workdir(REMOTE_DIR)
)
ollaya_image = _mount(
    modal.Image.from_registry(OLLAYA_IMAGE, add_python=PYTHON_VERSION)
    .entrypoint([])  # the image's ENTRYPOINT is the ollaya binary; Modal must exec python
    .pip_install(*RUNTIME_DEPS)
    .env(
        {
            "OLLAYA_HOST": OLLAYA_URL.split("//")[1],
            "OLLAYA_MODELS": OLLAYA_MODELS_DIR,
            "OLLAYA_DEVICE": "cuda",  # fail loudly rather than fall back to the CPU
            "OLLAYA_KEEP_ALIVE": "-1",
            "OLLAYA_LOAD_TIMEOUT": OLLAYA_LOAD_TIMEOUT,
        }
    )
    .workdir(REMOTE_DIR)
)
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
    answer: Callable[[DataPoint, int, list[Item]], int],
    reset: Callable[[], None] | None = None,
) -> int:
    """Every not-yet-answered batch of the shard's cases, committed to the Volume as it goes.

    `run.concurrency` client threads each drive one request at a time through `answer`, so that
    many requests are queued at the model together (Kev's Server batches whatever is queued)."""
    run = job.run
    if run.fanout:
        return _serve_fanout(job, answer, reset)
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


def _fanout_loop(
    job: ShardJob,
    q: modal.Queue,
    on_item: Callable[[tuple], None],
    wait_idle: Callable[[], None],
    reset: Callable[[], None] | None,
) -> None:
    """The worker protocol loop: get_many on the shard's partition.

    A work item is (case_id, bis, m). Control items: ("reset",) waits for in-flight work
    (`wait_idle`), calls `reset` if any, acks on "ack"; None stops the worker."""
    last_commit = time.time()
    while True:
        for item in q.get_many(1000, partition=f"w{job.shard}", block=True):
            if item is None:
                runs_volume.commit()
                return
            if item[0] == "reset":
                wait_idle()
                if reset is not None:
                    reset()
                q.put(("reset_ok", job.shard), partition="ack")
                continue
            on_item(item)
        if time.time() - last_commit > COMMIT_EVERY_S:
            runs_volume.commit()
            last_commit = time.time()


def _serve_fanout(
    job: ShardJob,
    answer: Callable[[DataPoint, int, list[Item]], int],
    reset: Callable[[], None] | None = None,
) -> int:
    """Fan-out worker: take (case_id, bis, m) items off the Queue, answer, signal on "done".

    The engine's `answer` closure is unchanged; `run.concurrency` threads queue requests at the
    model. The protocol is `_fanout_loop` (`reset` e.g. clears Kev's prefix cache)."""
    run = job.run
    out = run.shard_path(job.shard)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt"):
        pass
    q = modal.Queue.from_name(run.queue_name, create_if_missing=True)
    q.put(job.shard, partition="ready")
    cases = {c.id: c for c in load_cases()}
    n = 0

    def work(case: DataPoint, bi: int, m: int | None) -> None:
        nonlocal n
        try:
            batch = batches(case.input, m, run.max_chars)[bi]
            t0 = time.time()
            written = answer(case, bi, batch)
        except Exception as e:  # noqa: BLE001 - relayed to the dispatcher, which fails the run
            q.put(
                ["error", f"{case.id} batch {bi}: {type(e).__name__}: {e}", 1],
                partition="done",
            )
        else:
            n += 1
            q.put((case.id, bi, t0, time.time() - t0, written), partition="done")

    futs: set = set()
    with ThreadPoolExecutor(max_workers=run.concurrency, thread_name_prefix="client") as pool:

        def on_item(item: tuple) -> None:
            case = cases[item[0]]
            for bi in item[1]:
                futs.add(pool.submit(work, case, bi, item[2]))

        def wait_idle() -> None:
            nonlocal futs
            while futs:
                _, futs = wait(futs, timeout=1)

        _fanout_loop(job, q, on_item, wait_idle, reset)
    return n


def _fanout_pass_item(
    run: ScoringRun,
    case: DataPoint,
    bis: list[int],
    m: int | None,
    predict: Callable[[list[dict]], list[dict]],
) -> tuple[list[RawRecord], list[tuple]]:
    """One fan-out work item through a pass engine: `predict` in `run.forward_batch` chunks.

    Every request records its chunk's start and duration, exactly as `_answer_laya_shard` /
    `_answer_forward_shard` do; each done tuple is (case_id, batch, t0, worker_s, records=1)."""
    method: Method = METHODS[run.method]
    all_batches = batches(case.input, m, run.max_chars)
    todo = [(bi, all_batches[bi]) for bi in bis]
    records: list[RawRecord] = []
    dones: list[tuple] = []
    for lo in range(0, len(todo), run.forward_batch):
        chunk = todo[lo : lo + run.forward_batch]
        bodies = [request(method, case.input.query, batch, run.model).body() for _, batch in chunk]
        t = time.time()
        resps = predict(bodies)
        dt = time.time() - t
        for (bi, batch), resp in zip(chunk, resps, strict=True):
            resp_obj = SystemOneResponse.model_validate(resp)
            records.append(
                record(
                    method,
                    case.id,
                    bi,
                    batch,
                    resp_obj,
                    dt,
                    t,
                    resp_obj.usage.input_tokens or None,
                )
            )
            dones.append((case.id, bi, t, dt, 1))
    return records, dones


def _serve_fanout_passes(
    job: ShardJob,
    predict: Callable[[list[dict]], list[dict]],
) -> tuple[int, int]:
    """Fan-out worker for pass engines (Laya, RSI-Jev): same protocol as `_serve_fanout`, but a
    work item's requests run through `predict` in `run.forward_batch` chunks, in-line — pass
    engines batch inside one predict call, so no client pool is needed."""
    run = job.run
    out = run.shard_path(job.shard)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt"):
        pass
    q = modal.Queue.from_name(run.queue_name, create_if_missing=True)
    q.put(job.shard, partition="ready")
    cases = {c.id: c for c in load_cases()}
    n = passes = 0

    def on_item(item: tuple) -> None:
        nonlocal n, passes
        case = cases[item[0]]
        try:
            records, dones = _fanout_pass_item(run, case, item[1], item[2], predict)
        except Exception as e:  # noqa: BLE001 - relayed to the dispatcher, which fails the run
            q.put(
                [
                    "error",
                    f"{case.id} batches {item[1]}: {type(e).__name__}: {e}",
                    len(item[1]),
                ],
                partition="done",
            )
            return
        for rec in records:
            append_jsonl_gz(out, rec)
        for done_msg in dones:
            q.put(done_msg, partition="done")
        n += len(records)
        passes += (len(item[1]) + run.forward_batch - 1) // run.forward_batch

    _fanout_loop(job, q, on_item, lambda: None, None)
    return n, passes


def fanout_plan(requests_count: int, shards: int) -> list[list[int]]:
    """Per-worker request indices; request j goes to worker j % shards."""
    return [[j for j in range(requests_count) if j % shards == w] for w in range(shards)]


SWEEP_WARMUP = 3
SWEEP_REPS = 20
SWEEP_CASES = 4
DISPATCH_READY_TIMEOUT_S = 2 * HOUR
DISPATCH_WARMUP_CASES = 3


def _collect_done(
    q: modal.Queue, pending: int
) -> dict[tuple[str, int], tuple[float, float, float, int]]:
    """Wait for `pending` fan-out answers; (case_id, batch) -> (recv, worker_start, worker_s,
    records)."""
    got: dict[tuple[str, int], tuple[float, float, float, int]] = {}
    while len(got) < pending:
        items = q.get_many(1000, partition="done", block=True)
        now = time.time()
        for item in items:
            if item[0] == "error":
                raise RuntimeError(f"fan-out worker: {item[1]}")
            got[(item[0], item[1])] = (now, item[2], item[3], item[4])
    return got


def _reset_workers(run: ScoringRun, q: modal.Queue, pool: ThreadPoolExecutor) -> None:
    """Send every worker the reset control item and wait for all "ack"s (untimed handshake)."""
    list(pool.map(lambda w: q.put(("reset",), partition=f"w{w}"), range(run.shards)))
    got = 0
    while got < run.shards:
        got += len(q.get_many(1000, partition="ack", block=True))


def _await_workers(run: ScoringRun, q: modal.Queue) -> None:
    """Block until every worker has put its shard number on partition "ready" (2 h limit)."""
    deadline = time.time() + DISPATCH_READY_TIMEOUT_S
    got = 0
    with tqdm(total=run.shards, desc=f"{run.reranker} workers ready", unit="worker") as bar:
        while got < run.shards:
            try:
                items = q.get_many(1000, partition="ready", block=True, timeout=60)
            except queue.Empty:
                items = []
            got += len(items)
            bar.update(len(items))
            if got < run.shards and time.time() > deadline:
                raise TimeoutError(f"{run.reranker}: {got} of {run.shards} workers ready after 2h")


def _dispatch_case(
    run: ScoringRun,
    q: modal.Queue,
    case: DataPoint,
    warmup: bool,
    pool: ThreadPoolExecutor,
) -> list[dict]:
    """Reset the workers, send one case's requests over the pool, collect every answer.

    Requests are the original `batches(case.input, run.max_items, run.max_chars)` — fan-out
    only decides which GPU answers each one, never the request composition (answers depend
    on which children share a request)."""
    _reset_workers(run, q, pool)
    m = run.max_items
    n_requests = len(batches(case.input, m, run.max_chars))
    sent: dict[tuple[str, int], float] = {}

    def send(w: int, bis: list[int]) -> None:
        now = time.time()
        for bi in bis:
            sent[(case.id, bi)] = now
        q.put((case.id, bis, m), partition=f"w{w}")

    targets = [(w, b) for w, b in enumerate(fanout_plan(n_requests, run.shards)) if b]
    list(pool.map(lambda t: send(*t), targets))
    recv = _collect_done(q, n_requests)
    return [
        {
            "case_id": case.id,
            "batch": bi,
            "sent": sent[(case.id, bi)],
            "recv": recv[(case.id, bi)][0],
            "worker": bi % run.shards,
            "worker_start": recv[(case.id, bi)][1],
            "worker_s": round(recv[(case.id, bi)][2], 3),
            "records": recv[(case.id, bi)][3],
            "warmup": warmup,
        }
        for bi in range(n_requests)
    ]


def _dispatch_run(run: ScoringRun, q: modal.Queue, pool: ThreadPoolExecutor) -> dict:
    """3 untimed warm-up cases, a 2 s gap, then every case timed over the pool."""
    cases = load_cases()[: run.cases]
    rows = []
    for case in cases[:DISPATCH_WARMUP_CASES]:
        rows += _dispatch_case(run, q, case, True, pool)
    t_gap = time.time()
    time.sleep(2)
    timed_from = t_gap + 1.0
    with tqdm(cases, desc=f"{run.reranker} dispatch", unit="case", mininterval=5) as bar:
        for case in bar:
            rows += _dispatch_case(run, q, case, False, pool)
            wall = max(r["recv"] for r in rows if r["case_id"] == case.id) - min(
                r["sent"] for r in rows if r["case_id"] == case.id
            )
            bar.set_postfix_str(f"{wall:.2f}s")
    return {"timed_from": timed_from, "rows": rows}


def _collect_done_sweep(
    q: modal.Queue, pending: int
) -> tuple[dict[tuple[str, int], tuple[float, float, float, int]], str | None]:
    """Like `_collect_done` but tolerant: an ["error", msg, covered] item counts `covered`
    requests toward `pending` so the rep drains fully and no stale items leak into the next
    point. Returns the answers plus the first error message."""
    got: dict[tuple[str, int], tuple[float, float, float, int]] = {}
    covered = 0
    error = None
    while len(got) + covered < pending:
        items_ = q.get_many(1000, partition="done", block=True)
        now = time.time()
        for item in items_:
            if item[0] == "error":
                error = error or item[1]
                covered += item[2]
            else:
                got[(item[0], item[1])] = (now, item[2], item[3], item[4])
    return got, error


def _dispatch_sweep(run: ScoringRun, q: modal.Queue, pool: ThreadPoolExecutor) -> list[dict]:
    """Per sweep point: 3 warm-up + 20 timed reps, each rep one put to worker 0. A failing rep
    records `error` and ends that point; the sweep moves on to the next point."""
    cases = load_cases()[: run.cases]
    children_of = {c.id: len(items(c.input)) for c in cases}
    rows = []
    for pt in run.sweep or []:
        pool_cases = [c for c in cases if children_of[c.id] >= pt.requests * pt.children]
        if not pool_cases:
            raise ValueError(
                f"{run.reranker} sweep {pt.requests}x{pt.children}: "
                f"no case with >= {pt.requests * pt.children} children"
            )
        pool_cases = pool_cases[:SWEEP_CASES]
        for rep in range(SWEEP_WARMUP + SWEEP_REPS):
            case = pool_cases[rep % len(pool_cases)]
            sent_batches = batches(case.input, pt.children, run.max_chars)[: pt.requests]
            _reset_workers(run, q, pool)
            t_send = time.time()
            q.put((case.id, list(range(len(sent_batches))), pt.children), partition="w0")
            recv, error = _collect_done_sweep(q, len(sent_batches))
            row = {
                "requests": pt.requests,
                "children": pt.children,
                "rep": rep,
                "case_id": case.id,
                "children_sent": sum(len(b) for b in sent_batches),
                "warmup": rep < SWEEP_WARMUP,
            }
            rows.append(row)
            if error is not None:
                row["error"] = error
                break
            row["records"] = sum(r[3] for r in recv.values())
            row["wall_s"] = round(max(r[0] for r in recv.values()) - t_send, 3)
    return rows


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

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> int:
        req = request(method, case.input.query, batch, "kev-latest")
        kev_req = SystemOneRequest.model_validate(req.body())  # our contract -> kev.api's
        t = time.time()
        try:
            body = server.answer(kev_req)
        except HTTPException as e:
            if e.status_code != 422 or len(batch) == 1:  # 422 = over Kev's state token limit
                raise
            half = len(batch) // 2
            return answer(case, bi, batch[:half]) + answer(case, bi, batch[half:])
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
        return 1

    def reset() -> None:
        # Fan-out queries are new states: a prefix-cache hit is only legitimate inside a query.
        with server.lock:
            server.prefix_cache.clear()

    n = _answer_shard(job, done, answer, reset)
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

    predict = functools.partial(laya_batch.predict_batch, agent)
    if run.fanout:
        n, passes = _serve_fanout_passes(job, predict)
    else:
        n, passes = _answer_laya_shard(job, done, predict)
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

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> int:
        body = request(method, case.input.query, batch, run.model).body()
        t = time.time()
        resp = SystemOneResponse.model_validate(
            joint_schema_model.systemone(model, processor, body)
        )
        latency_s = time.time() - t
        rec = record(method, case.id, bi, batch, resp, latency_s, t, resp.usage.input_tokens)
        append_jsonl_gz(out, rec)
        return 1

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

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> int:
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
        return 1

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

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> int:
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
        return 1

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

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> int:
        body = request(method, case.input.query, batch, served).body()
        t = time.time()
        try:
            resp = SystemOneResponse.model_validate(jev(body))
        except ValueError as e:
            if "exceeds" not in str(e) or len(batch) == 1:
                raise
            half = len(batch) // 2
            return answer(case, bi, batch[:half]) + answer(case, bi, batch[half:])
        latency_s = time.time() - t
        rec = record(method, case.id, bi, batch, resp, latency_s, t, resp.usage.input_tokens)
        append_jsonl_gz(out, rec)
        return 1

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

    if run.fanout:
        n, _forward_passes = _serve_fanout_passes(job, predict)
    else:
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

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> int:
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
        return 1

    n = _answer_shard(job, done, answer)
    return _summary(job, t0, loaded, n, len(done), resident_gb)


def _post_answer(
    client: httpx.Client,
    url: str,
    out: Path,
    method: Method,
    model: str,
    write: threading.Lock,
    case: DataPoint,
    bi: int,
    batch: list[Item],
) -> int:
    """Post one batch's System One request to `url` and append the record to the shard file.

    A batch the endpoint rejects as over its input limit (400/422) is split in two, as
    score_cases does on Kev's 422. A transient error is retried with exponential backoff; a
    record's latency is its successful attempt."""
    body = request(method, case.input.query, batch, model).body()
    for attempt in range(API_RETRIES + 1):
        t = time.time()
        r = client.post(url, json=body)
        if r.status_code not in API_TRANSIENT or attempt == API_RETRIES:
            break
        print(f"{case.id} batch {bi}: HTTP {r.status_code}, retry {attempt + 1}", flush=True)
        time.sleep(min(API_BACKOFF_S * 2**attempt, API_MAX_WAIT_S))
    if r.status_code in (400, 422) and len(batch) > 1:
        half = len(batch) // 2
        return _post_answer(
            client, url, out, method, model, write, case, bi, batch[:half]
        ) + _post_answer(client, url, out, method, model, write, case, bi, batch[half:])
    if r.status_code != 200:
        raise RuntimeError(f"{case.id} batch {bi}: HTTP {r.status_code}: {r.text[:300]}")
    rec = record(
        method, case.id, bi, batch, SystemOneResponse.model_validate(r.json()), time.time() - t, t
    )
    with write:
        append_jsonl_gz(out, rec)
    return 1


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

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> int:
        body = request(method, case.input.query, batch, run.model).body()
        t = time.time()
        answers, usage = model.decide(body["state"], body["questions"])
        resp = SystemOneResponse.model_validate(
            {"model": run.model, "answers": answers, "usage": usage}
        )
        latency_s = time.time() - t
        rec = record(method, case.id, bi, batch, resp, latency_s, t, usage["input_tokens"])
        append_jsonl_gz(out, rec)
        return 1

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

    def answer(case: DataPoint, bi: int, batch: list[Item]) -> int:
        body = request(method, case.input.query, batch, run.model).body()
        t = time.time()
        result = engine.evaluate(state=body["state"], questions=body["questions"], model=run.model)
        resp = SystemOneResponse.model_validate(von.response(run.model, body["questions"], result))
        latency_s = time.time() - t
        rec = record(method, case.id, bi, batch, resp, latency_s, t, resp.usage.input_tokens)
        append_jsonl_gz(out, rec)
        return 1

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

    `run.concurrency` requests in flight (see _post_answer for splitting and retries)."""
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
    client = httpx.Client(headers=headers, timeout=API_TIMEOUT)
    answer = functools.partial(
        _post_answer, client, run.api.url, out, method, run.model, threading.Lock()
    )
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


def _start_llama_server(model_path: str, parallel: int) -> subprocess.Popen:
    """llama-server at LLAMA_URL with every layer on the GPU; returns once /health answers 200."""
    cmd = [
        f"{LLAMA_CPP_DIR}/llama-server",
        *("--model", model_path),
        *("--host", LLAMA_URL.split("//")[1].split(":")[0]),
        *("--port", LLAMA_URL.rsplit(":", 1)[1]),
        *("--n-gpu-layers", "999"),
        *("--parallel", str(parallel)),
        *("--ctx-size", str(LLAMA_CTX_PER_SLOT * parallel)),
        "--no-webui",
    ]
    print(" ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=LLAMA_LOG.open("w"), stderr=subprocess.STDOUT)
    started = time.time()
    while time.time() - started < LLAMA_START_TIMEOUT_S and proc.poll() is None:
        try:
            if httpx.get(f"{LLAMA_URL}/health", timeout=LLAMA_POLL_S).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(LLAMA_POLL_S)
    proc.kill()
    raise RuntimeError(
        f"llama-server not healthy after {time.time() - started:.0f}s (exit {proc.poll()}):\n"
        + LLAMA_LOG.read_text()[-3000:]
    )


@app.function(
    image=gguf_image,
    gpu=DEFAULT_GGUF_GPU,
    timeout=4 * HOUR,
    single_use_containers=True,
    volumes=VOLUMES,
)
def score_cases_gguf(job_json: str) -> str:
    """Answer every batch with a GGUF decision model behind llama.cpp's /v1/systemone; resumable.

    `run.file` of the Hub repo `run.model` (repo@revision) is downloaded into the HF-cache Volume
    (tqdm; reused next time), llama-server serves it on localhost with `run.concurrency` slots,
    and the API posting loop is pointed at it, so the shard files are those of every scorer and
    kept-mass, warm GPU-seconds, $/1k and latency need no new code. Load time = download + server
    start + one warm-up request (the mmap'd weights are paged in by the first forward pass, not
    by /health); warm time = the posting loop."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    job = ShardJob.model_validate_json(job_json)
    run = job.run
    if run.file is None:
        raise ValueError(f"{job.desc}: gguf source without a file")
    method = METHODS[run.method]
    out = run.shard_path(job.shard)
    done = {(r.case_id, r.batch) for r in read_jsonl_gz(out)}
    t0 = time.time()
    repo, _, revision = run.model.partition("@")
    try:
        path = hf_hub_download(repo, run.file, revision=revision or None)
    except EntryNotFoundError as e:
        raise FileNotFoundError(f"{job.desc}: no file {run.file} in {run.model}") from e
    hf_cache.commit()
    server = _start_llama_server(path, run.concurrency)
    client = httpx.Client(timeout=API_TIMEOUT)
    url = f"{LLAMA_URL}/v1/systemone"
    answer = functools.partial(_post_answer, client, url, out, method, run.model, threading.Lock())
    try:
        cases = run.shard_cases(job.shard)
        if cases:
            first = batches(cases[0].input, run.max_items, run.max_chars)[0]
            body = request(method, cases[0].input.query, first, run.model).body()
            client.post(url, json=body).raise_for_status()
        loaded = time.time()
        n = _answer_shard(job, done, answer)
    finally:
        client.close()
        server.terminate()
        print(LLAMA_LOG.read_text()[-2500:], flush=True)
    scored = time.time()
    return ShardSummary(
        reranker=run.reranker,
        shard=job.shard,
        gpu=run.gpu_type,
        load_s=loaded - t0,
        warm_s=scored - loaded,
        total_s=time.time() - t0,
        requests=n,
        resumed=len(done),
    ).model_dump_json()


def _start_ollaya() -> subprocess.Popen:
    """The Ollaya daemon at OLLAYA_URL (env from ollaya_image); returns once GET / answers 200."""
    Path(OLLAYA_MODELS_DIR).mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [OLLAYA_BIN, "serve"], stdout=OLLAYA_LOG.open("w"), stderr=subprocess.STDOUT
    )
    started = time.time()
    while time.time() - started < OLLAYA_START_TIMEOUT_S and proc.poll() is None:
        try:
            if httpx.get(OLLAYA_URL, timeout=LLAMA_POLL_S).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(LLAMA_POLL_S)
    proc.kill()
    raise RuntimeError(
        f"ollaya serve not up after {time.time() - started:.0f}s (exit {proc.poll()}):\n"
        + OLLAYA_LOG.read_text()[-3000:]
    )


def pull_progress(lines: Iterable[str], bars: dict[str, tqdm]) -> None:
    """Drive one tqdm bar per blob from /api/pull's NDJSON stream (`pulling <digest>` lines carry
    total and completed bytes); the stream must end in {"status": "success"}, else it was cut."""
    last: dict | None = None
    for line in lines:
        if not line:
            continue
        last = json.loads(line)
        if "error" in last:
            raise RuntimeError(f"ollaya pull: {last['error']}")
        digest = last.get("digest")
        if digest is None:
            continue
        if digest not in bars:
            bars[digest] = tqdm(
                total=last["total"],
                desc=f"pull {digest[7:19]}",
                unit="B",
                unit_scale=True,
                mininterval=PROGRESS_EVERY_S,
            )
        bars[digest].update(last["completed"] - bars[digest].n)
    if last is None or last.get("status") != "success":
        raise RuntimeError(f"ollaya pull: stream ended without success (last line {last})")


def _pull_ollaya_model(client: httpx.Client, model: str) -> None:
    """`ollaya pull`: the ONNX graph and the upstream weight files it points at, into the store."""
    bars: dict[str, tqdm] = {}
    with client.stream(
        "POST", f"{OLLAYA_URL}/api/pull", json={"model": model}, timeout=OLLAYA_PULL_TIMEOUT
    ) as r:
        if r.status_code != 200:
            r.read()
            raise RuntimeError(f"ollaya pull {model}: HTTP {r.status_code}: {r.text[:300]}")
        try:
            pull_progress(r.iter_lines(), bars)
        finally:
            for bar in bars.values():
                bar.close()


@app.function(
    image=ollaya_image,
    gpu=DEFAULT_OLLAYA_GPU,
    timeout=4 * HOUR,
    single_use_containers=True,
    volumes=VOLUMES,
)
def score_cases_ollaya(job_json: str) -> str:
    """Answer every batch with an ONNX decision model behind Ollaya's /v1/systemone; resumable.

    `run.model` is an Ollaya name (kev:9b, laya:en, ...): the daemon pulls its ONNX graph and the
    upstream weight files into the store on the HF-cache Volume (tqdm; reused next time) and runs
    it with ONNX Runtime CUDA. Load time = pull + daemon start + one warm-up request (which loads
    the model onto the GPU); warm time = the posting loop, as for every API-shaped scorer."""
    job = ShardJob.model_validate_json(job_json)
    run = job.run
    method = METHODS[run.method]
    out = run.shard_path(job.shard)
    done = {(r.case_id, r.batch) for r in read_jsonl_gz(out)}
    t0 = time.time()
    server = _start_ollaya()
    client = httpx.Client(timeout=API_TIMEOUT)
    url = f"{OLLAYA_URL}/v1/systemone"
    answer = functools.partial(_post_answer, client, url, out, method, run.model, threading.Lock())
    try:
        _pull_ollaya_model(client, run.model)
        hf_cache.commit()
        cases = run.shard_cases(job.shard)
        if cases:
            first = batches(cases[0].input, run.max_items, run.max_chars)[0]
            body = request(method, cases[0].input.query, first, run.model).body()
            r = client.post(url, json=body, timeout=OLLAYA_PULL_TIMEOUT)
            if r.status_code != 200:
                raise RuntimeError(f"{job.desc}: warm-up HTTP {r.status_code}: {r.text[:300]}")
        loaded = time.time()
        print(f"loaded {run.model} in {loaded - t0:.0f}s", flush=True)
        n = _answer_shard(job, done, answer)
    finally:
        client.close()
        server.terminate()
        print(OLLAYA_LOG.read_text()[-2500:], flush=True)
    scored = time.time()
    return ShardSummary(
        reranker=run.reranker,
        shard=job.shard,
        gpu=run.gpu_type,
        load_s=loaded - t0,
        warm_s=scored - loaded,
        total_s=time.time() - t0,
        requests=n,
        resumed=len(done),
    ).model_dump_json()


@app.function(image=api_image, timeout=4 * HOUR, single_use_containers=True, volumes=VOLUMES)
def dispatch_cases(job_json: str) -> str:
    """Fan-out dispatcher: wait for the workers, then drive the case pass or the sweep.

    Runs on CPU on the light api image; timing (sent / recv) is client-side, so a query's
    wall clock is the same shape as the prod/Jev API timing."""
    job = ShardJob.model_validate_json(job_json)
    run = job.run
    q = modal.Queue.from_name(run.queue_name, create_if_missing=True)
    _await_workers(run, q)
    try:
        with ThreadPoolExecutor(max_workers=min(run.shards, 64)) as send_pool:
            if run.sweep:
                _write_remote_json(
                    run.remote_dir / "sweep.json", _dispatch_sweep(run, q, send_pool)
                )
            else:
                _write_remote_json(
                    run.remote_dir / "dispatch.json", _dispatch_run(run, q, send_pool)
                )
    finally:
        for w in range(run.shards):
            q.put(None, partition=f"w{w}")
    runs_volume.commit()
    return f"{run.reranker}: dispatched"


def _write_remote_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1) + "\n")


SCORERS = {
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
    "gguf": score_cases_gguf,
    "ollaya": score_cases_ollaya,
}


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
    if run.fanout:
        _record_call(
            ShardJob(run=run, shard=0, role="dispatch"),
            dispatch_cases.spawn(ShardJob(run=run, shard=0, role="dispatch").model_dump_json()),
            calls_file,
        )


def _record_call(job: ShardJob, call: modal.FunctionCall, calls_file: Path) -> None:
    rec = CallRecord(
        job=job,
        call_id=call.object_id,
        spawned_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    calls_file.parent.mkdir(parents=True, exist_ok=True)
    with calls_file.open("a") as f:
        f.write(rec.model_dump_json() + "\n")
    print(f"spawned {call.object_id}: {job.desc}")


def spawn_shard(job: ShardJob, calls_file: Path) -> None:
    """Spawn (or re-spawn: done batches on the Volume are skipped) one shard and record its call."""
    run = job.run
    options: dict = {"memory": MEMORY_MB_FOR.get(run.repo)}
    if run.engine == "api":
        if run.api is not None and run.api.secret is not None:
            options["secrets"] = [modal.Secret.from_name(run.api.secret)]
    else:
        options["gpu"] = run.gpu_type
    fn = SCORERS[run.engine].with_options(**options)
    call = fn.spawn(job.model_dump_json())
    _record_call(job, call, calls_file)
    print(f"  ({len(run.shard_cases(job.shard))} cases)")


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
    entries = [e for e in entries if e.path.endswith(".jsonl.gz")]
    if len(entries) != run.shards:
        raise FileNotFoundError(f"{remote}: {len(entries)} of {run.shards} shard files")
    records: list[RawRecord] = []
    for entry in tqdm(entries, desc=f"pull {run.reranker}", unit="shard"):
        with io.BytesIO(b"".join(vol.read_file(entry.path))) as buf, gzip.open(buf, "rt") as f:
            records += [RawRecord.model_validate_json(line) for line in f if line.strip()]
    records.sort(key=lambda r: (r.case_id, r.batch))
    return records


def pull_json(run: ScoringRun, name: str) -> dict | list:
    """A JSON file the dispatcher wrote next to the shard files (dispatch.json / sweep.json)."""
    vol = modal.Volume.from_name(RUNS_VOLUME_NAME)
    remote = run.remote_dir.relative_to(RUNS_ROOT) / name
    return json.loads(b"".join(vol.read_file(str(remote))))

"""R-1 timers: what one GPU shard spent (load vs warm), what a reranker cost, and per-query wall clock.

Warm = from the loaded server to the last answer; that is what a long-lived container pays per query.
A shard whose call failed has no summary (None) and is counted as unmeasured, never dropped.
"""

import re
from collections.abc import Iterable

from pydantic import BaseModel

from jev_tracker.systemone import RawRecord

# Shard return value before the timers were split: "..., load 36s, total 940s on H100 (...)".
_LEGACY = re.compile(r"load (\d+)s, total (\d+)s on ([\w-]+)")


class Marks(BaseModel):
    """time.time() at container start, server loaded, last answer, function return."""

    model_config = {"frozen": True}

    start: float
    loaded: float
    scored: float
    end: float

    @property
    def load_s(self) -> float:
        return self.loaded - self.start

    @property
    def warm_s(self) -> float:
        return self.scored - self.loaded

    @property
    def total_s(self) -> float:
        return self.end - self.start


class ShardSummary(BaseModel):
    """One shard's return value (modal_app.score_cases), as JSON."""

    model_config = {"frozen": True}

    reranker: str
    shard: int
    gpu: str  # the billed (requested) GPU type
    load_s: float
    warm_s: float
    total_s: float  # in-function: load + warm + teardown
    requests: int | None  # None = a legacy summary that did not count them
    device: str | None = None  # the GPU actually attached
    resident_gb: float | None = None
    peak_gb: float | None = None  # max allocated while scoring
    forward_passes: int | None = None
    graphs: dict[str, int] | None = None  # kev CudaGraphs.stats() at the end
    option_isolation: bool | None = None  # vLLM/SGLang shards: the checkpoint's flag (H-2, H-3)
    resumed: int = 0  # requests already on the Volume when this call started; warm_s excludes them


class CostRow(BaseModel):
    model_config = {"frozen": True}

    gpu: str | None
    shards: int
    unmeasured: int
    resumed: int = 0  # requests the latest calls skipped: warm_gpu_s doesn't cover them
    load_s: float
    warm_gpu_s: float
    in_function_s: float
    warm_usd: float
    in_function_usd: float
    # None when any measured shard's summary predates the field (legacy text summaries)
    requests: int | None = None
    forward_passes: int | None = None
    resident_gb: float | None = None  # max over shards
    peak_gb: float | None = None  # max over shards


def _sum(xs: list[int | None]) -> int | None:
    return None if not xs or None in xs else sum(x for x in xs if x is not None)


def _max(xs: list[float | None]) -> float | None:
    return None if not xs or None in xs else max(x for x in xs if x is not None)


def parse_summary(state: str) -> ShardSummary | None:
    """A finished call's return value -> its summary; None for a failed or still-running call."""
    if state.startswith("{"):
        return ShardSummary.model_validate_json(state)
    m = _LEGACY.search(state)
    if m is None:
        return None
    load_s, total_s = float(m[1]), float(m[2])
    return ShardSummary(
        reranker="",
        shard=-1,
        gpu=m[3],
        load_s=load_s,
        warm_s=total_s - load_s,
        total_s=total_s,
        requests=None,
    )


def shard_costs(summaries: Iterable[ShardSummary | None], usd_per_s: float) -> CostRow:
    """One reranker's shards -> GPU-seconds and $ at `usd_per_s` (GPU + host RAM) per shard-second."""
    rows = list(summaries)
    ok = [s for s in rows if s is not None]
    warm = sum(s.warm_s for s in ok)
    total = sum(s.total_s for s in ok)
    return CostRow(
        gpu=ok[0].gpu if ok else None,
        shards=len(rows),
        unmeasured=len(rows) - len(ok),
        resumed=sum(s.resumed for s in ok),
        load_s=sum(s.load_s for s in ok),
        warm_gpu_s=warm,
        in_function_s=total,
        warm_usd=warm * usd_per_s,
        in_function_usd=total * usd_per_s,
        requests=_sum([s.requests for s in ok]),
        forward_passes=_sum([s.forward_passes for s in ok]),
        resident_gb=_max([s.resident_gb for s in ok]),
        peak_gb=_max([s.peak_gb for s in ok]),
    )


def query_wall_s(records: Iterable[RawRecord]) -> dict[str, float]:
    """Per case: first request sent to last answer back (overlapping requests are not summed)."""
    spans: dict[str, tuple[float, float]] = {}
    for r in records:
        if r.started_at_s is None:
            raise ValueError(f"{r.case_id} batch {r.batch}: no started_at_s, wall clock unknown")
        lo, hi = spans.get(r.case_id, (r.started_at_s, r.started_at_s))
        spans[r.case_id] = (min(lo, r.started_at_s), max(hi, r.started_at_s + r.latency_s))
    return {c: round(hi - lo, 3) for c, (lo, hi) in spans.items()}

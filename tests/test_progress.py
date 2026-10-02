"""The per-shard progress a Modal worker publishes for the site's Running cards."""

from pathlib import Path

from jev_tracker.experiment import ModelSource, load_config
from jev_tracker.modal_app import PROGRESS_EVERY_S, ProgressBar, ScoringRun, ShardJob, ShardProgress

CONFIG = Path("configs/kev08b_vs_jev.yaml")


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeStore:
    def __init__(self) -> None:
        self.puts: list[tuple[str, dict]] = []

    def put(self, key: str, value: dict) -> None:
        self.puts.append((key, value))


def job(shards: int = 2, shard: int = 1) -> ShardJob:
    run = ScoringRun(
        config=CONFIG.as_posix(),
        experiment="2026_10_02_00_00_00_tidy_otter",
        reranker="kev08b_score",
        engine="kev",
        model="jaredpalmer/kev-0.8b",
        method="score",
        max_items=25,
        max_chars=24_000,
        shards=shards,
    )
    return ShardJob(run=run, shard=shard)


def test_run_config_reaches_the_scoring_run_so_the_site_can_match_the_queue_item() -> None:
    exp = load_config(CONFIG)
    name, src = next((n, s) for n, s in exp.rerankers.items() if isinstance(s, ModelSource))
    assert exp.scoring_run("x", name, src, CONFIG.as_posix()).config == "configs/kev08b_vs_jev.yaml"
    assert exp.scoring_run("x", name, src).config is None  # old calls.jsonl records still parse


def test_bar_publishes_at_start_then_at_most_every_interval_and_at_the_end() -> None:
    clock, store = FakeClock(), FakeStore()
    bar = ProgressBar(job(), resumed=3, total=10, store=store, clock=clock)  # type: ignore[arg-type]
    assert [p[1]["done"] for p in store.puts] == [3]  # the resumed count, before any answer

    clock.now += 1
    bar.update(4)
    bar.update(5)
    assert len(store.puts) == 1  # inside the interval: nothing new on the wire

    clock.now += PROGRESS_EVERY_S
    bar.update(6)
    assert [p[1]["done"] for p in store.puts] == [3, 6]

    clock.now += 1
    bar.update(10)  # the last request always publishes, interval or not
    assert [p[1]["done"] for p in store.puts] == [3, 6, 10]

    key, last = store.puts[-1]
    assert key == "2026_10_02_00_00_00_tidy_otter/kev08b_score/01"
    published = ShardProgress.model_validate(last)
    assert published.finished
    assert published.config == CONFIG.as_posix()
    assert (published.shard, published.shards, published.total, published.resumed) == (1, 2, 10, 3)
    assert published.started_at == 1000.0
    assert published.updated_at == clock.now


def test_unfinished_bar_is_not_finished() -> None:
    store = FakeStore()
    ProgressBar(job(), resumed=0, total=4, store=store, clock=FakeClock())  # type: ignore[arg-type]
    assert not ShardProgress.model_validate(store.puts[-1][1]).finished

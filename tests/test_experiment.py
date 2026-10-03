"""Experiment helpers: billing rates."""

from jev_tracker.experiment import GPU_USD_PER_S, RAM_USD_PER_GIB_S, _usd_per_s


def test_pinned_model_bills_ram_like_the_unpinned_repo() -> None:
    pinned = _usd_per_s("H100", "jaredpalmer/kev-27b@01b8199")
    unpinned = _usd_per_s("H100", "jaredpalmer/kev-27b")
    assert pinned == unpinned == GPU_USD_PER_S["H100"] + 128 * RAM_USD_PER_GIB_S

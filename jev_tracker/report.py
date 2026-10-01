"""The deliverable: one markdown table (kept-mass@k + win-rate) and one kept-mass@k curve."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jev_tracker.kept_mass import Table  # noqa: E402


def markdown(means: Table, wins: Table, n_cases: int) -> str:
    ks = sorted(next(iter(means.values())))
    head = "| ranker | " + " | ".join(f"@{k}" for k in ks) + " |\n|---|" + "---:|" * len(ks) + "\n"
    out = f"kept-mass@k, mean over {n_cases} labeled cases\n\n" + head
    out += "".join(
        f"| {m} | " + " | ".join(f"{v[k]:.3f}" for k in ks) + " |\n" for m, v in means.items()
    )
    out += "\nwin-rate vs prod (ties = 1/2)\n\n" + head
    out += "".join(
        f"| {m} | " + " | ".join(f"{v[k]:.2f}" for k in ks) + " |\n" for m, v in wins.items()
    )
    return out


BASELINES = ("oracle", "random")
# Okabe-Ito palette (distinguishable under all common color-vision deficiencies), and a
# distinct marker + line style per series so the curves are legible in grayscale too.
COLORS = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#0072B2",
    "#F0E442",
    "#000000",
    "#999999",
    "#882255",
    "#44AA99",
    "#332288",
)
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p")
LINES = ("-", "--", "-.", ":")
assert len(COLORS) == len(MARKERS)


def plot(means: Table, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ranked = [m for m in means if m not in BASELINES]
    for m, v in means.items():
        ks = sorted(v)
        if m in BASELINES:
            style = {
                "linestyle": "--" if m == "oracle" else ":",
                "color": "#7F7F7F",
                "marker": "o",
                "markersize": 4,
            }
        else:
            i = ranked.index(m)
            style = {
                "color": COLORS[i % len(COLORS)],
                "marker": MARKERS[i % len(MARKERS)],
                "linestyle": LINES[i % len(LINES)],
                "markersize": 7,
                "markeredgecolor": "black",
                "markeredgewidth": 0.5,
            }
        ax.plot(ks, [v[k] for k in ks], label=m, linewidth=1.8, **style)
    ax.set_xlabel("k")
    ax.set_ylabel("kept-mass@k (mean over cases)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def write_report(means: Table, wins: Table, n_cases: int, base: Path) -> str:
    """<base>.{md,png,json}; returns the markdown."""
    base.parent.mkdir(parents=True, exist_ok=True)
    md = markdown(means, wins, n_cases)
    base.with_suffix(".md").write_text(md)
    base.with_suffix(".json").write_text(
        json.dumps({"n_cases": n_cases, "kept_mass": means, "win_rate_vs_prod": wins}, indent=1)
    )
    plot(means, base.with_suffix(".png"))
    print(f"wrote {base}.{{md,json,png}}")
    return md

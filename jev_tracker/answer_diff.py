"""Per-child answer comparison of two raw answer files.

    uv run python -m jev_tracker.answer_diff A.json.gz B.json.gz [--json out.json]

Keys on (case_id, parent_index, child_index, option) and reports |a-b| stats over the
intersection. The noise floor for identical request composition is ~0.01 at max."""

import argparse
import json
from pathlib import Path

from jev_tracker.rerankers import read_raw


def child_answers(path: Path) -> dict[tuple, float]:
    """(case_id, parent_index, child_index, option) -> the child's numeric answer."""
    out: dict[tuple, float] = {}
    for r in read_raw(path):
        for s in r.scores:
            a = r.answers.get(s.question)
            if a is None:
                continue
            v = getattr(a, a.type)
            if isinstance(v, dict):
                v = v.get(s.option) if s.option is not None else None
            if isinstance(v, (int, float)):
                out[(r.case_id, s.parent_index, s.child_index, s.option)] = v
    return out


def diff(a_path: Path, b_path: Path) -> dict:
    a, b = child_answers(a_path), child_answers(b_path)
    ds = sorted(abs(a[k] - b[k]) for k in set(a) & set(b))
    n = len(ds)
    q = lambda p: ds[min(n - 1, int(p * n))] if n else None  # noqa: E731
    return {
        "a": str(a_path),
        "b": str(b_path),
        "n": n,
        "n_a": len(a),
        "n_b": len(b),
        "mean": sum(ds) / n if n else None,
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": ds[-1] if n else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    stats = diff(args.a, args.b)
    if args.json:
        args.json.write_text(json.dumps(stats, indent=1) + "\n")
    print(
        f"n={stats['n']} (a={stats['n_a']}, b={stats['n_b']}) |a-b| "
        f"mean={stats['mean']:.6f} p50={stats['p50']} p90={stats['p90']} "
        f"p99={stats['p99']} max={stats['max']}"
    )


if __name__ == "__main__":
    main()

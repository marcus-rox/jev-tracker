"""kept-mass@k and win-rate-vs-prod over the labeled cases (k = 50..200 step 10 by default).

Verbatim metric from reranker_evals/jev_simple_ranking_1/kept_mass_eval.py (PR #324):
  kept-mass@k = sum(rating of the ranker's top k) / sum(rating of the best possible top k),
                k capped at the case's labeled children; 1.0 when the case has no rating mass.
  win-rate@k  = fraction of cases where ranker's kept-mass@k > prod's (ties count 1/2).
Only labeled children take part; ties in a ranker's scores break by production order, as before.
Pure evaluation: rerankers come in as `Reranker`s (see experiment.py for where they come from).
"""

import random
import statistics

from pydantic import BaseModel

from jev_tracker.contract import ChildRef, DataPoint, load_cases, load_labels
from jev_tracker.rerankers import Reranker

RANDOM_SEED = 0
PROD = "prod"  # the row win-rates are measured against
PerCase = dict[str, dict[int, list[float]]]  # ranker -> k -> kept-mass per case
Table = dict[str, dict[int, float]]  # ranker -> k -> value


class LabeledCase(BaseModel):
    """One case's labeled children in production order, with their ratings."""

    model_config = {"frozen": True}

    id: str
    refs: list[ChildRef]
    ratings: list[float]


def labeled_cases(
    cases: list[DataPoint], ratings: dict[tuple[str, ChildRef], float]
) -> list[LabeledCase]:
    out = []
    for c in cases:
        refs = [r for r in c.production_output.ranking if (c.id, r) in ratings]
        out.append(LabeledCase(id=c.id, refs=refs, ratings=[ratings[(c.id, r)] for r in refs]))
    return out


def kept_mass(case: LabeledCase, scores: list[float], k: int) -> float:
    k = min(k, len(case.ratings))
    top = sorted(range(len(case.ratings)), key=lambda i: -scores[i])[:k]
    best = sum(sorted(case.ratings, reverse=True)[:k])
    return round(sum(case.ratings[i] for i in top) / best, 9) if best else 1.0


def oracle_scores(case: LabeledCase) -> list[float]:
    return list(case.ratings)


def random_scores(case: LabeledCase, rng: random.Random) -> list[float]:
    order = list(range(len(case.ratings)))
    rng.shuffle(order)
    return [float(x) for x in order]


def evaluate(rerankers: list[Reranker], ks: list[int], cases_n: int | None) -> tuple[PerCase, int]:
    """ranker -> k -> per-case kept-mass, plus the case count. Baselines oracle/random included."""
    cases = load_cases()[:cases_n]
    ratings = {
        (lp.case_id, ChildRef(parent_index=lp.parent_index, child_index=lp.child_index)): lp.rating
        for lp in load_labels()
    }
    labeled = labeled_cases(cases, ratings)
    by_id = {c.id: c for c in cases}
    rng = random.Random(RANDOM_SEED)
    per_case: dict[str, list[list[float]]] = {
        "oracle": [oracle_scores(c) for c in labeled],
        "random": [random_scores(c, rng) for c in labeled],
    }
    for r in rerankers:
        per_case[r.name] = [[r.scores(by_id[c.id])[ref] for ref in c.refs] for c in labeled]
    km = {
        name: {k: [kept_mass(c, s, k) for c, s in zip(labeled, rows, strict=True)] for k in ks}
        for name, rows in per_case.items()
    }
    return km, len(labeled)


def win_rate(mine: list[float], prod: list[float]) -> float:
    pairs = zip(mine, prod, strict=True)
    return sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in pairs) / len(mine)


def tables(km: PerCase, ks: list[int]) -> tuple[Table, Table]:
    """Mean kept-mass@k per ranker, and win-rate@k vs PROD (empty if PROD was not evaluated)."""
    means = {m: {k: statistics.mean(v) for k, v in per_k.items()} for m, per_k in km.items()}
    wins = (
        {m: {k: win_rate(km[m][k], km[PROD][k]) for k in ks} for m in km if m != PROD}
        if PROD in km
        else {}
    )
    return means, wins

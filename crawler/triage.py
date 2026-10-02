"""R-9: Devin's decision per crawler candidate, recorded as data in the PR.

    crawler/triage/<YYYY-MM-DD>.yaml          a YAML list of TriageDecision, one per candidate
    python -m crawler.triage check crawler/triage/<date>.yaml crawler/candidates/<date>.jsonl

`check` exits 1 listing candidates with no decision and `runnable` decisions whose config file
does not exist (paths relative to the repo root); otherwise it prints counts per verdict.
Decisions are matched to candidates by `url` until `Candidate.key` lands (see candidate_key).
"""

import argparse
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from crawler.contract import Candidate

REPO_DIR = Path(__file__).resolve().parents[1]
Verdict = Literal["runnable", "needs_adapter", "not_jev"]
VERDICTS: tuple[Verdict, ...] = ("runnable", "needs_adapter", "not_jev")


class TriageDecision(BaseModel):
    model_config = {"frozen": True}

    key: str
    url: str
    verdict: Verdict
    reason: str
    config: Path | None = None  # configs/<name>.yaml written for a runnable candidate


def candidate_key(candidate: Candidate) -> str:
    return candidate.url


def read_triage(path: Path) -> list[TriageDecision]:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, list):
        raise ValueError(f"{path}: expected a YAML list of decisions, got {type(loaded).__name__}")
    return [TriageDecision.model_validate(item) for item in loaded]


def write_triage(path: Path, decisions: list[TriageDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dumped = [decision.model_dump(mode="json") for decision in decisions]
    path.write_text(yaml.safe_dump(dumped, sort_keys=False, allow_unicode=True))


def read_candidates(path: Path) -> list[Candidate]:
    return [Candidate.model_validate_json(line) for line in path.read_text().splitlines() if line]


def undecided(candidates: list[Candidate], decisions: list[TriageDecision]) -> list[str]:
    """Candidate keys with no decision, in candidate order."""
    decided = {decision.key for decision in decisions}
    return [candidate_key(c) for c in candidates if candidate_key(c) not in decided]


def missing_configs(decisions: list[TriageDecision], root: Path) -> list[TriageDecision]:
    """Runnable decisions whose config is unset or does not exist under `root`."""
    return [
        decision
        for decision in decisions
        if decision.verdict == "runnable"
        and (decision.config is None or not (root / decision.config).exists())
    ]


def check(triage_path: Path, candidates_path: Path, root: Path) -> int:
    decisions = read_triage(triage_path)
    candidates = read_candidates(candidates_path)
    no_decision = undecided(candidates, decisions)
    no_config = missing_configs(decisions, root)
    for key in no_decision:
        print(f"no decision: {key}")
    for decision in no_config:
        print(f"runnable without config: {decision.key} (config: {decision.config})")
    if no_decision or no_config:
        print(f"{len(no_decision)} undecided, {len(no_config)} runnable without config")
        return 1
    counts = "  ".join(
        f"{verdict} {sum(d.verdict == verdict for d in decisions)}" for verdict in VERDICTS
    )
    print(f"{len(decisions)} decisions for {len(candidates)} candidates: {counts}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m crawler.triage", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check")
    check_parser.add_argument("triage", type=Path, help="crawler/triage/<date>.yaml")
    check_parser.add_argument("candidates", type=Path, help="crawler/candidates/<date>.jsonl")
    check_parser.add_argument(
        "--root", type=Path, default=REPO_DIR, help="configs are relative to it"
    )
    args = parser.parse_args(argv)
    return check(args.triage, args.candidates, args.root)


if __name__ == "__main__":
    sys.exit(main())

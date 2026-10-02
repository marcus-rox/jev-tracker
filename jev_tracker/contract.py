"""Types and loaders for the 75-case parent-child reranker golden set (frozen copy from PR #324).

Input = one query + k parents, parent i having x_i children. Output = one global order of all
children, best first. A LabeledPair rates cases[i].input.parents[parent_index].children[child_index].
"""

import json
from pathlib import Path

from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmark"
CASES_FILE = DATA_DIR / "golden_eval_75.json"
LABELS_FILE = DATA_DIR / "golden_labels_75.jsonl"


class Parent(BaseModel):
    model_config = {"frozen": True}

    text: str
    children: list[str]


class RerankInput(BaseModel):
    model_config = {"frozen": True}

    query: str
    parents: list[Parent]


class ChildRef(BaseModel):
    model_config = {"frozen": True}

    parent_index: int
    child_index: int


class RerankOutput(BaseModel):
    model_config = {"frozen": True}

    ranking: list[ChildRef]  # every child exactly once, best first


class DataPoint(BaseModel):
    model_config = {"frozen": True}

    id: str
    input: RerankInput
    production_output: RerankOutput


class LabeledPair(BaseModel):
    model_config = {"frozen": True}

    case_id: str
    query: str
    parent_index: int
    child_index: int
    parent: str
    child: str
    votes: list[int]  # 3 grader votes, each 0..3
    rating: float  # mean of votes


def load_cases(path: Path = CASES_FILE) -> list[DataPoint]:
    return [DataPoint.model_validate(d) for d in json.loads(path.read_text())]


def load_labels(path: Path = LABELS_FILE) -> list[LabeledPair]:
    with path.open() as f:
        return [LabeledPair.model_validate_json(line) for line in f if line.strip()]

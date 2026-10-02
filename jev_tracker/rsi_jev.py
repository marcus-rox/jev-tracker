"""RSI-Jev's wire mapping: a System One request body -> the release's typed questions, and one
option distribution -> one System One answer.

Ported from github.com/Shanghua-Gao/RSI-Jev serve/wire.py (main on 2026-10-02, 1a30a277): the
state is rendered as compact JSON (a string passes through), noul's options are ("false", "true")
with "No" / "Yes" defaults, a choice's options are its keys in order, a score's are its level
indices, and `confidence` is (K * p_max - 1) / (K - 1). Serving does not cut a state; the release
answers up to 32,768 tokens per question. Pure Python, so it is tested without a GPU.
"""

import json
from typing import Literal

from pydantic import BaseModel

MAX_INPUT_TOKENS = 32_768
NOUL_OPTIONS = ("false", "true")
NOUL_DEFAULTS = {"true": "Yes", "false": "No"}


class RsiQuestion(BaseModel):
    """The release's `rsijev.contract.Question`, field for field."""

    model_config = {"frozen": True}

    key: str
    mode: Literal["choice", "noul", "score"]
    instructions: str
    options: tuple[str, ...]
    criteria: dict[str, str]  # option -> its description


def serialize(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def state_text(body: dict) -> str:
    return serialize(body["state"])


def question(key: str, spec: dict) -> RsiQuestion:
    criteria = spec.get("criteria")
    instructions = serialize(spec["instructions"])
    if spec["type"] == "noul":
        given = criteria or {}
        described = {
            option: serialize(given.get(option, NOUL_DEFAULTS[option])) for option in NOUL_OPTIONS
        }
        return RsiQuestion(
            key=key,
            mode="noul",
            instructions=instructions,
            options=NOUL_OPTIONS,
            criteria=described,
        )
    if spec["type"] == "choice":
        described = {
            option: "" if text is None else serialize(text) for option, text in criteria.items()
        }
        return RsiQuestion(
            key=key,
            mode="choice",
            instructions=instructions,
            options=tuple(criteria),
            criteria=described,
        )
    levels = tuple(str(level) for level in range(len(criteria)))
    described = {str(level): serialize(text) for level, text in enumerate(criteria)}
    return RsiQuestion(
        key=key, mode="score", instructions=instructions, options=levels, criteria=described
    )


def questions(body: dict) -> list[RsiQuestion]:
    return [question(key, spec) for key, spec in body["questions"].items()]


def confidence(probs: list[float]) -> float:
    count = len(probs)
    return min(1.0, max(0.0, (count * max(probs) - 1) / (count - 1)))


def answer(asked: RsiQuestion, probs: list[float]) -> dict:
    total = sum(probs)
    probs = [prob / total for prob in probs]
    if asked.mode == "noul":
        return {"type": "noul", "noul": probs[1]}
    distribution = dict(zip(asked.options, probs, strict=True))
    if asked.mode == "score":
        return {
            "type": "score",
            "score": sum(level * prob for level, prob in enumerate(probs)),
            "confidence": confidence(probs),
            "probabilities": distribution,
        }
    best = max(range(len(probs)), key=probs.__getitem__)
    return {
        "type": "choice",
        "choice": asked.options[best],
        "confidence": confidence(probs),
        "probabilities": distribution,
    }

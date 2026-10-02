"""AutoTrust JEV-27B's decision prompt (protocol jev27-bare-v1) for System One requests.

The release (huggingface.co/autotrust/JEV-27B, README "Plain transformers + peft") answers one
{kind, state, question, options} at a time: the bare prompt below, the merged-LoRA Qwen3.8-27B
backbone's last hidden state through the 24-slot decision head (`head.safetensors`), the kind's
slots over its calibrated temperature (`calibration.json`), softmax. This module is the torch-free
half: System One request -> prompts, head logits -> System One response.

A System One question maps onto it as: noul -> noul (false / true; criteria appended to the
question), choice -> choice over its criteria (A-P), score -> choice over its levels (the release's
own score kind is a fixed 0-5 scale the levels' text would not reach)."""

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

LETTERS = "ABCDEFGHIJKLMNOP"  # the head's 16 choice slots
NOUL_OPTIONS = ["false", "true"]
HEAD_FILES = ["judge_config.json", "calibration.json"]


class Head(BaseModel):
    """Slot ranges per kind (judge_config.json) and per-kind temperatures (calibration.json)."""

    model_config = {"frozen": True}

    ranges: dict[str, tuple[int, int]]
    temperatures: dict[str, float]

    @classmethod
    def load(cls, path: Path) -> "Head":
        slots = json.loads((path / "judge_config.json").read_text())["slots"]
        calibration = json.loads((path / "calibration.json").read_text())
        return cls(ranges=slots["ranges"], temperatures=calibration["per_kind"])


class Prompt(BaseModel):
    model_config = {"frozen": True}

    key: str  # the question's key in the request
    question_type: Literal["noul", "choice", "score"]
    kind: Literal["noul", "choice"]  # the release's kind it is asked as
    options: list[str]  # answer keys in slot order: false/true, criteria keys or level indexes
    text: str


def _text(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _prompt(
    key: str,
    question_type: Literal["noul", "choice", "score"],
    state: str,
    question: str,
    options: list[str],
    lines: list[str],
) -> Prompt:
    kind = "noul" if question_type == "noul" else "choice"
    if kind == "choice":
        if not 2 <= len(lines) <= len(LETTERS):
            raise ValueError(f"{key}: {len(lines)} options; the head has 2-{len(LETTERS)}")
        lines = [f"{LETTERS[index]}) {line}" for index, line in enumerate(lines)]
    text = (
        f"[kind] {kind}\n[state] {state}\n[question] {question}\n[options]\n"
        + "\n".join(lines)
        + "\n[decision]:"
    )
    return Prompt(key=key, question_type=question_type, kind=kind, options=options, text=text)


def prompts(body: dict) -> list[Prompt]:
    """One release prompt per question of a /v1/systemone request body."""
    state = _text(body["state"])
    out = []
    for key, question in body["questions"].items():
        instructions = _text(question.get("instructions") or "")
        if question["type"] == "noul":
            criteria = question.get("criteria")
            if criteria:
                instructions += f"\ntrue: {criteria['true']}\nfalse: {criteria['false']}"
            out.append(_prompt(key, "noul", state, instructions, NOUL_OPTIONS, NOUL_OPTIONS))
        elif question["type"] == "choice":
            criteria = question["criteria"]
            lines = [f"{name}: {_text(text)}" if text else name for name, text in criteria.items()]
            out.append(_prompt(key, "choice", state, instructions, list(criteria), lines))
        else:
            levels = [_text(level) for level in question["criteria"]]
            options = [str(index) for index in range(len(levels))]
            out.append(_prompt(key, "score", state, instructions, options, levels))
    return out


def probabilities(prompt: Prompt, logits: list[float], head: Head) -> list[float]:
    """Softmax over the prompt's slots of the 24 head logits, at the kind's temperature."""
    low, _ = head.ranges[prompt.kind]
    temperature = head.temperatures[prompt.kind]
    scaled = [logits[low + index] / temperature for index in range(len(prompt.options))]
    top = max(scaled)
    weights = [math.exp(value - top) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def answer(prompt: Prompt, probs: list[float]) -> dict:
    best = max(range(len(probs)), key=probs.__getitem__)
    distribution = dict(zip(prompt.options, probs, strict=True))
    if prompt.question_type == "noul":
        return {"type": "noul", "noul": probs[1]}
    if prompt.question_type == "choice":
        return {
            "type": "choice",
            "choice": prompt.options[best],
            "confidence": probs[best],
            "probabilities": distribution,
        }
    return {
        "type": "score",
        "score": sum(index * prob for index, prob in enumerate(probs)),
        "confidence": probs[best],
        "probabilities": distribution,
    }


def response(
    model: str, asked: list[Prompt], logits: dict[str, list[float]], head: Head, input_tokens: int
) -> dict:
    """The System One response body for one request's prompts and their head logits."""
    return {
        "model": model,
        "answers": {
            prompt.key: answer(prompt, probabilities(prompt, logits[prompt.key], head))
            for prompt in asked
        },
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }

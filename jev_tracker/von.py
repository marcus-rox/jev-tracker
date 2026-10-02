"""Von's wire mapping: a System One request body -> the SDK's `VonEngine.evaluate` arguments,
and its `SystemOneResponse` -> ours.

github.com/wfzyx/von (von-sdk): `engine.evaluate(state, questions, model)` takes the request's
questions as raw {"type", "instructions", "criteria"} dicts and answers every one. Von's answer
models carry no `type` field, so the contract's discriminator is put back from the question.
Its `score` answers' probabilities are keyed by level index, matching our score answers.
"""

from typing import Any


def answers(questions: dict[str, Any], result_answers: dict[str, Any]) -> dict[str, dict]:
    """Von's answers plus the `type` each question asked for (its models drop the field)."""
    return {
        key: {"type": questions[key]["type"], **answer.model_dump()}
        for key, answer in result_answers.items()
    }


def response(model: str, questions: dict[str, Any], result: Any) -> dict:
    """A von `SystemOneResponse` as a plain dict for our contract to validate."""
    return {
        "model": model,
        "answers": answers(questions, result.answers),
        "usage": result.usage.model_dump(),
    }

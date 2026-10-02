"""MiniCPM5-2B-Jev's answers: the release's per-question distributions -> System One answers.

The release (huggingface.co/ytbai/MiniCPM5-2B-Jev, model.py) converts the request body itself
(`to_internal_record`) and returns one distribution per question over that question's `keys`
(noul: "false", "true"; score: "0" .. "n-1"); this maps them as its serve.py does, plus the
System One readouts (noul = p(true), score = sum(i * p_i)). Pure Python, tested without a GPU.
"""


def answer(question_type: str, keys: list[str], probs: list[float]) -> dict:
    total = sum(probs)
    probs = [prob / total for prob in probs]
    distribution = dict(zip(keys, probs, strict=True))
    best = max(range(len(probs)), key=probs.__getitem__)
    if question_type == "noul":
        return {"type": "noul", "noul": distribution["true"]}
    if question_type == "score":
        return {
            "type": "score",
            "score": sum(level * prob for level, prob in enumerate(probs)),
            "confidence": probs[best],
            "probabilities": distribution,
        }
    return {
        "type": "choice",
        "choice": keys[best],
        "confidence": probs[best],
        "probabilities": distribution,
    }


def response(
    model: str, internal_questions: list[dict], probs: list[list[float]], input_tokens: int
) -> dict:
    """The System One response body for one request: `internal_questions` is the release's
    `to_internal_record(body)["questions"]`, `probs` its distributions in the same order."""
    return {
        "model": model,
        "answers": {
            asked["name"]: answer(asked["type"], asked["keys"], question_probs)
            for asked, question_probs in zip(internal_questions, probs, strict=True)
        },
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }

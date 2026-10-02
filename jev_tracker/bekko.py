"""Bekko System One v0's wire mapping: a System One request body -> one predict() input object,
and one decision's prediction -> one System One answer.

huggingface.co/hotchpotch/bekko-system-one-v0-{17m,68m,400m} (inference_v0): predict takes
{"state_json": <json>, "decisions": [{id, kind: "judgment", type, instructions_json: <json>,
system_prompt: "", criteria: [{id, description_json: <json>, value}], documents: [], scoring:
None}]} and returns, keyed by decision id, {"selected_id", "probabilities"} (choice),
{"probability_yes", "probabilities"} (noul; ids "true"/"false" or "yes"/"no"), or {"score",
"normalized_score", "probabilities", "values"} (score; each criterion carries a numeric value).
Our criteria keep their names: a choice's option keys, noul's "false"/"true", a score's level
index as both id and value (so its probabilities double as our level distribution). `confidence`
is (K * p_max - 1) / (K - 1), as rsi_jev. Pure Python, so it is tested without a GPU.
"""

import json

NOUL_IDS = ("false", "true")
NOUL_DEFAULTS = {"true": "Yes", "false": "No"}


def serialize(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def criterion(id: str, description: object, value: float | None) -> dict:
    return {"id": id, "description_json": serialize(description), "value": value}


def decision(key: str, spec: dict) -> dict:
    criteria = spec.get("criteria")
    if spec["type"] == "noul":
        given = criteria or {}
        options = [
            criterion(option, given.get(option, NOUL_DEFAULTS[option]), None) for option in NOUL_IDS
        ]
    elif spec["type"] == "choice":
        options = [criterion(option, text, None) for option, text in criteria.items()]
    else:
        options = [criterion(str(level), text, float(level)) for level, text in enumerate(criteria)]
    return {
        "id": key,
        "kind": "judgment",
        "type": spec["type"],
        "instructions_json": serialize(spec["instructions"]),
        "system_prompt": "",
        "criteria": options,
        "documents": [],
        "scoring": None,
    }


def input_object(body: dict) -> dict:
    """One request body as one predict() input: shared state, one judgment per question."""
    return {
        "state_json": serialize(body["state"]),
        "decisions": [decision(key, spec) for key, spec in body["questions"].items()],
    }


def confidence(probs: list[float]) -> float:
    count = len(probs)
    return min(1.0, max(0.0, (count * max(probs) - 1) / (count - 1)))


def answer(spec: dict, prediction: dict) -> dict:
    if spec["type"] == "noul":
        return {"type": "noul", "noul": prediction["probability_yes"]}
    probs = prediction["probabilities"]
    if spec["type"] == "score":
        return {
            "type": "score",
            "score": prediction["score"],
            "confidence": confidence(list(probs.values())),
            "probabilities": probs,
        }
    return {
        "type": "choice",
        "choice": prediction["selected_id"],
        "confidence": confidence(list(probs.values())),
        "probabilities": probs,
    }

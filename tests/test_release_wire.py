import math

from jev_tracker import minicpm_jev, rsi_jev
from jev_tracker.methods import METHODS, Item, request
from jev_tracker.systemone import SystemOneResponse

BATCH = [
    Item(parent_index=0, child_index=0, text="Paris is the capital of France."),
    Item(parent_index=0, child_index=1, text="Cats purr."),
]


def body(method: str, model: str) -> dict:
    return request(METHODS[method], "capital of France?", BATCH, model).body()


def test_rsi_jev_state_is_compact_json_with_the_query_last() -> None:
    noul = body("noul_query_in_state", "shgao/rsi-jev-v3.0-qwen3.5-2b")
    state = rsi_jev.state_text(noul)
    assert state.startswith('{"items":{"i0":"Paris is the capital of France."')
    assert state.endswith(',"query":"capital of France?"}')


def test_rsi_jev_noul_defaults_and_readout() -> None:
    noul = body("noul_query_in_state", "shgao/rsi-jev-v3.0-qwen3.5-2b")
    asked = rsi_jev.questions(noul)
    assert [q.key for q in asked] == list(noul["questions"])
    assert asked[0].options == ("false", "true")
    assert asked[0].criteria == {"false": "No", "true": "Yes"}
    assert rsi_jev.answer(asked[0], [0.25, 0.75]) == {"type": "noul", "noul": 0.75}


def test_rsi_jev_score_levels_are_indices_and_score_is_expected_level() -> None:
    score = body("score_query_in_question", "shgao/rsi-jev-v3.0-qwen3.5-2b")
    asked = rsi_jev.questions(score)
    levels = len(next(iter(score["questions"].values()))["criteria"])
    assert asked[0].options == tuple(str(level) for level in range(levels))
    probs = [0.0] * levels
    probs[-1] = 1.0
    answers = {q.key: rsi_jev.answer(q, probs) for q in asked}
    resp = SystemOneResponse.model_validate(
        {"model": "rsi", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}}
    )
    assert resp.answers[asked[0].key].score == levels - 1
    assert resp.answers[asked[0].key].confidence == 1.0


def test_rsi_jev_confidence_is_peak_above_chance() -> None:
    assert math.isclose(rsi_jev.confidence([0.5, 0.25, 0.25, 0.0]), 1 / 3)
    assert rsi_jev.confidence([0.5, 0.5]) == 0.0


def test_minicpm_jev_response_maps_keys_to_answers() -> None:
    internal = [
        {"name": "i0", "type": "noul", "keys": ["false", "true"]},
        {"name": "i1", "type": "score", "keys": ["0", "1", "2"]},
    ]
    resp = SystemOneResponse.model_validate(
        minicpm_jev.response("minicpm", internal, [[0.2, 0.8], [0.1, 0.1, 0.8]], 42)
    )
    assert resp.answers["i0"].noul == 0.8
    assert math.isclose(resp.answers["i1"].score, 1.7)
    assert resp.answers["i1"].probabilities == {"0": 0.1, "1": 0.1, "2": 0.8}
    assert resp.usage.input_tokens == 42

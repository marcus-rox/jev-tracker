import math

from jev_tracker import autotrust
from jev_tracker.methods import METHODS, Item, request
from jev_tracker.systemone import SystemOneResponse

HEAD = autotrust.Head(
    ranges={"noul": (0, 2), "score": (2, 8), "choice": (8, 24)},
    temperatures={"noul": 1.0, "score": 1.0, "choice": 2.0},
)
BATCH = [
    Item(parent_index=0, child_index=0, text="Paris is the capital of France."),
    Item(parent_index=0, child_index=1, text="Cats purr."),
]


def body(method: str) -> dict:
    return request(METHODS[method], "capital of France?", BATCH, "autotrust/JEV-27B").body()


def test_noul_prompt_is_the_release_bare_prompt() -> None:
    prompts = autotrust.prompts(body("noul_query_in_state"))
    assert [p.key for p in prompts] == list(body("noul_query_in_state")["questions"])
    first = prompts[0]
    assert first.kind == "noul" and first.options == ["false", "true"]
    assert first.text.startswith('[kind] noul\n[state] {"items": {')
    assert first.text.endswith("[options]\nfalse\ntrue\n[decision]:")


def test_score_is_asked_as_a_choice_over_its_levels() -> None:
    prompt = autotrust.prompts(body("score_query_in_question"))[0]
    assert prompt.kind == "choice" and prompt.question_type == "score"
    assert prompt.options == ["0", "1", "2", "3"]
    assert "\nA) Not relevant: " in prompt.text and "\nD) Directly relevant: " in prompt.text


def answered(method: str, logits: list[float]) -> SystemOneResponse:
    prompts = autotrust.prompts(body(method))
    return SystemOneResponse.model_validate(
        autotrust.response("autotrust/JEV-27B", prompts, {p.key: logits for p in prompts}, HEAD, 10)
    )


def test_response_reads_the_kind_slots_at_its_temperature() -> None:
    logits = [0.0] * 24
    logits[1] = math.log(3.0)  # true vs false: 3:1
    logits[8 + 3] = 2 * math.log(2.0)  # level 3 at choice temperature 2 -> weight 2
    noul = answered("noul_query_in_state", logits)
    assert all(math.isclose(a.noul, 0.75) for a in noul.answers.values())
    level = next(iter(answered("score_query_in_question", logits).answers.values()))
    assert math.isclose(level.probabilities["3"], 0.4)
    assert math.isclose(level.score, (0 + 1 + 2) * 0.2 + 3 * 0.4)

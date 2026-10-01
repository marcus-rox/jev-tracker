"""How a System One decision model (Jev, Kev, CLM, ...) is asked about a case's children.

A Method turns (query, named children) into one request's `state` + `questions` and says where
each child's answer is (`Binding`: which question, and for a choice which option), and how that
answer becomes a float. `batches` packs a case's children into requests; `request` builds one,
`record` stores its response with the bindings (both typed by systemone.py, so every model is
asked and answered identically).

Two shapes exist: one question per child (`noul_query_in_state`, `score_query_in_question`,
verbatim from the Jev run this project reproduces, so Jev's committed raw answers stay valid;
`noul_query_in_question` and `noul_query_in_question_options` reword noul in the single-field
yes/no style of Laya's presets, the query in the question, optionally with yes/no option text) and
one choice question over all children (`choice_over_children`: the state is the query, the
options are the children, so a dual-encoder model such as CLM answers it as cosine similarity).
"""

from collections.abc import Callable

from pydantic import BaseModel

from jev_tracker.contract import RerankInput
from jev_tracker.systemone import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulCriteria,
    NoulQuestion,
    Question,
    RawRecord,
    RawScore,
    ScoreAnswer,
    ScoreQuestion,
    State,
    SystemOneRequest,
    SystemOneResponse,
)

JEV_MAX_ITEMS, JEV_MAX_CHARS = 75, 120_000  # production's batch limits, used for the Jev runs

RELEVANCE_LEVELS = [
    "Not relevant: nothing in the answer relates to the query",
    "Slightly relevant: touches the topic but does not help answer the query",
    "Mostly relevant: contains part of what the query asks for",
    "Directly relevant: contains the core of what the query asks for",
]
NOUL_SEARCH_OPTIONS = NoulCriteria(
    true="the passage helps answer the search", false="the passage does not help answer the search"
)
CHOICE_QUESTION_ID = "best"


class Item(BaseModel):
    model_config = {"frozen": True}

    parent_index: int
    child_index: int
    text: str


class Binding(BaseModel):
    """Where a child's answer is in a response."""

    model_config = {"frozen": True}

    question: str
    option: str | None = None


class Ask(BaseModel):
    """One request's content and, per item name, where its answer will be."""

    model_config = {"frozen": True}

    state: State
    questions: dict[str, Question]
    bindings: dict[str, Binding]


class Method(BaseModel):
    model_config = {"frozen": True}

    name: str
    ask: Callable[[str, dict[str, str]], Ask]  # (query, item name -> text) -> request content
    score: Callable[[Answer, Binding], float]  # higher = more relevant


def _per_child(
    state: Callable[[str, dict[str, str]], State], question: Callable[[str, str], Question]
) -> Callable[[str, dict[str, str]], Ask]:
    """One question per item, keyed by the item's name and asking about `items.<name>`."""

    def ask(query: str, named: dict[str, str]) -> Ask:
        return Ask(
            state=state(query, named),
            questions={k: question(f"items.{k}", query) for k in named},
            bindings={k: Binding(question=k) for k in named},
        )

    return ask


def _choice_over_children(query: str, named: dict[str, str]) -> Ask:
    return Ask(
        state=State(query=query),
        questions={
            CHOICE_QUESTION_ID: ChoiceQuestion(
                type="choice",
                instructions="Which of these answers the query?",
                criteria=named,
            )
        },
        bindings={k: Binding(question=CHOICE_QUESTION_ID, option=k) for k in named},
    )


def _p_yes(answer: Answer, binding: Binding) -> float:
    if not isinstance(answer, NoulAnswer):
        raise TypeError(f"noul method got a {answer.type!r} answer: {answer}")
    return answer.noul


def _normalized_expected_level(answer: Answer, binding: Binding) -> float:
    if not isinstance(answer, ScoreAnswer):
        raise TypeError(f"score method got a {answer.type!r} answer: {answer}")
    probs = answer.probabilities
    return sum(int(i) * p for i, p in probs.items()) / (len(probs) - 1)


def _p_option(answer: Answer, binding: Binding) -> float:
    if not isinstance(answer, ChoiceAnswer) or binding.option is None:
        raise TypeError(f"choice method got a {answer.type!r} answer for {binding}")
    return answer.probabilities[binding.option]


NOUL_QUERY_IN_STATE = Method(
    name="noul_query_in_state",
    ask=_per_child(
        lambda query, named: State(query=query, items=named),
        lambda ref, q: NoulQuestion(
            type="noul", instructions=f"Is `{ref}` a relevant answer to `query`?"
        ),
    ),
    score=_p_yes,
)


def _noul_search(criteria: NoulCriteria | None) -> Callable[[str, str], Question]:
    return lambda ref, q: NoulQuestion(
        type="noul",
        instructions=f"Does `{ref}` contain information that answers this search: {q}?",
        criteria=criteria,
    )


NOUL_QUERY_IN_QUESTION = Method(
    name="noul_query_in_question",
    ask=_per_child(lambda query, named: State(items=named), _noul_search(None)),
    score=_p_yes,
)

NOUL_QUERY_IN_QUESTION_OPTIONS = Method(
    name="noul_query_in_question_options",
    ask=_per_child(lambda query, named: State(items=named), _noul_search(NOUL_SEARCH_OPTIONS)),
    score=_p_yes,
)

SCORE_QUERY_IN_QUESTION = Method(
    name="score_query_in_question",
    ask=_per_child(
        lambda query, named: State(items=named),
        lambda ref, q: ScoreQuestion(
            type="score",
            instructions=f"The answer `{ref}` is relevant to: {q}",
            criteria=RELEVANCE_LEVELS,
        ),
    ),
    score=_normalized_expected_level,
)

CHOICE_OVER_CHILDREN = Method(
    name="choice_over_children", ask=_choice_over_children, score=_p_option
)

METHODS: dict[str, Method] = {
    m.name: m
    for m in (
        NOUL_QUERY_IN_STATE,
        NOUL_QUERY_IN_QUESTION,
        NOUL_QUERY_IN_QUESTION_OPTIONS,
        SCORE_QUERY_IN_QUESTION,
        CHOICE_OVER_CHILDREN,
    )
}


def items(inp: RerankInput) -> list[Item]:
    """Every child of a case, parents-then-children order, as the text the model sees."""
    return [
        Item(parent_index=pi, child_index=ci, text=f"In {p.text}: {c}")
        for pi, p in enumerate(inp.parents)
        for ci, c in enumerate(p.children)
    ]


def batches(inp: RerankInput, max_items: int | None, max_chars: int | None) -> list[list[Item]]:
    """Consecutive children per request; None = no limit (the whole case in one request)."""
    out: list[list[Item]] = []
    cur: list[Item] = []
    chars = 0
    for it in items(inp):
        full = (max_items is not None and len(cur) >= max_items) or (
            max_chars is not None and chars + len(it.text) > max_chars
        )
        if cur and full:
            out.append(cur)
            cur, chars = [], 0
        cur.append(it)
        chars += len(it.text)
    if cur:
        out.append(cur)
    return out


def item_name(n: int) -> str:
    return f"i{n}"


def named(batch: list[Item]) -> dict[str, str]:
    return {item_name(n): it.text for n, it in enumerate(batch)}


def request(method: Method, query: str, batch: list[Item], model: str) -> SystemOneRequest:
    """One /v1/systemone request for a batch; items are named `i0..iN` in batch order."""
    a = method.ask(query, named(batch))
    return SystemOneRequest(state=a.state, model=model, questions=a.questions)


def record(
    method: Method,
    case_id: str,
    batch_index: int,
    batch: list[Item],
    resp: SystemOneResponse,
    latency_s: float,
    started_at_s: float | None = None,
    state_tokens: int | None = None,
) -> RawRecord:
    """Store a response with each child bound to its answer. Model-agnostic."""
    bindings = method.ask("", named(batch)).bindings
    missing = sorted({b.question for b in bindings.values()} - set(resp.answers))
    if missing:
        raise ValueError(f"{case_id} batch {batch_index}: no answer for {missing}")
    return RawRecord(
        case_id=case_id,
        batch=batch_index,
        model=resp.model,
        usage=resp.usage,
        latency_s=round(latency_s, 3),
        started_at_s=started_at_s,
        state_tokens=state_tokens,
        answers=resp.answers,
        scores=[
            RawScore(
                parent_index=it.parent_index,
                child_index=it.child_index,
                question=bindings[item_name(n)].question,
                option=bindings[item_name(n)].option,
            )
            for n, it in enumerate(batch)
        ],
    )

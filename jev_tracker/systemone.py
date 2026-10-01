"""The one request/answer contract every decision model in this project speaks.

This is the /v1/systemone wire shape (Jev's API; Kev's `kev.api` and CLM's `clm.Engine.answer`
re-implement it). A model is *only* ever seen through these types:

    request  = SystemOneRequest(state, model, questions)      -> the model
    response = SystemOneResponse(model, answers, usage)       <- the model
    stored   = RawRecord(case_id, batch, ..., answers, scores=[RawScore(parent, child, question, option)])

`methods.request` builds the request, a producer (Jev over HTTP, `modal_app.score_cases` for Kev,
`modal_app.score_cases_clm` for CLM) returns the response, `methods.record` stores it with a
binding from each child to the answer (and option) that is about it, and the evaluator
(`rerankers.ScoredReranker`) scores with `Method.score`. Nothing downstream of `record` can tell
which model answered.

Question/answer variants are the three System One types, discriminated on `type`. Fields a model
adds beyond the contract (Kev/CLM echo the score `legend`, CLM adds `billing_units`) are dropped
at validation.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

FROZEN = {"frozen": True}


class State(BaseModel):
    """What the model reads before answering: the candidate texts, optionally the query."""

    model_config = FROZEN

    items: dict[str, str] = {}  # item name (`i0`, `i1`, ...) -> text the model sees
    query: str | None = None


class NoulCriteria(BaseModel):
    model_config = FROZEN

    true: str  # option text for "yes"
    false: str  # option text for "no"


class NoulQuestion(BaseModel):
    model_config = FROZEN

    type: Literal["noul"]
    instructions: str
    criteria: NoulCriteria | None = None  # None = the model's own yes/no option text


class ChoiceQuestion(BaseModel):
    model_config = FROZEN

    type: Literal["choice"]
    instructions: str
    criteria: dict[str, str] = Field(min_length=1)  # option key -> option text


class ScoreQuestion(BaseModel):
    model_config = FROZEN

    type: Literal["score"]
    instructions: str
    criteria: list[str] = Field(min_length=2)  # level 0 .. n-1, worst to best


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model_config = FROZEN

    state: State
    model: str
    questions: dict[str, Question] = Field(min_length=1)  # question id -> question

    def body(self) -> dict[str, object]:
        """JSON body as sent on the wire; `state` carries only the fields the method set."""
        return self.model_dump(exclude_none=True, exclude_defaults=True)


class NoulAnswer(BaseModel):
    model_config = FROZEN

    type: Literal["noul"]
    noul: float = Field(ge=0, le=1)  # P(yes)


class ChoiceAnswer(BaseModel):
    model_config = FROZEN

    type: Literal["choice"]
    choice: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]  # option key -> probability


class ScoreAnswer(BaseModel):
    model_config = FROZEN

    type: Literal["score"]
    score: float = Field(ge=0)  # expected level, sum(i * p_i)
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]  # level index as string -> probability


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(BaseModel):
    model_config = FROZEN

    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model_config = FROZEN

    model: str
    answers: dict[str, Answer]  # question id -> answer; same keys as the request's questions
    usage: Usage


class RawScore(BaseModel):
    """Where one child's answer is: the question about it and, for a choice, its option key."""

    model_config = FROZEN

    parent_index: int
    child_index: int
    question: str  # key into RawRecord.answers
    option: str | None = None  # key into a ChoiceAnswer's probabilities


class RawRecord(BaseModel):
    """One request's response for one case, one JSONL line. Same shape for every model."""

    model_config = FROZEN

    case_id: str
    batch: int
    model: str
    usage: Usage
    latency_s: float
    answers: dict[str, Answer]  # the response's answers, verbatim
    scores: list[RawScore]  # one per child in the request
    started_at_s: float | None = None  # time.time() when the request was sent; None in older runs
    state_tokens: int | None = None  # Kev's encoded state length (graph-bank admission); Kev only


class RawFile(BaseModel):
    """A reranker's complete answers for one experiment, one committed JSON file."""

    model_config = FROZEN

    run: str  # what produced the answers, e.g. kev-4b_noul_query_in_state_i25_c24000
    written_at: str
    records: list[RawRecord]

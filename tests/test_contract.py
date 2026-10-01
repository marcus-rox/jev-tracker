"""R-2: every decision model in jev-tracker speaks one contract (jev_tracker/systemone.py).

Free, offline: validates the committed Jev and Kev answers through the shared types, checks the
request builder against Kev's own `kev.api` schema (vendored field list, kev is not a dependency),
and that the evaluator's view is identical whichever model produced the answers.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from jev_tracker.contract import load_cases
from jev_tracker.experiment import LayaSource, Paths, load_config, new_id
from jev_tracker.methods import (
    CHOICE_QUESTION_ID,
    METHODS,
    Binding,
    batches,
    record,
    request,
)
from jev_tracker.rerankers import ScoredReranker, read_raw
from jev_tracker.systemone import (
    ChoiceAnswer,
    NoulAnswer,
    RawFile,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT = "2026_09_26_00_26_02_picked-bee"
RAW = {
    ("jev", "noul_query_in_state"): REPO / "data/jev/noul_query_in_state.json.gz",
    ("jev", "score_query_in_question"): REPO / "data/jev/score_query_in_question.json.gz",
    ("kev", "noul_query_in_state"): REPO
    / f"data/experiments/{EXPERIMENT}/raw_kev4b_noul_{EXPERIMENT}.json.gz",
    ("kev", "score_query_in_question"): REPO
    / f"data/experiments/{EXPERIMENT}/raw_kev4b_score_{EXPERIMENT}.json.gz",
}
ANSWER_TYPE = {"noul_query_in_state": NoulAnswer, "score_query_in_question": ScoreAnswer}
# kev.api.SystemOneRequest at modal_app.KEV_REF: the fields it declares and the question types.
KEV_REQUEST_FIELDS = {"state", "model", "questions"}
KEV_QUESTION_TYPES = {"noul", "choice", "score"}


@pytest.mark.parametrize(("model", "method"), sorted(RAW))
def test_committed_answers_validate_through_shared_contract(model: str, method: str) -> None:
    records = read_raw(RAW[model, method])
    assert records
    for rec in records:
        assert len(rec.scores) == len(rec.answers)
        for s in rec.scores:
            assert s.option is None
            assert type(rec.answers[s.question]) is ANSWER_TYPE[method], (model, rec.case_id, s)


@pytest.mark.parametrize("method", sorted(ANSWER_TYPE))
def test_jev_and_kev_answers_are_the_same_type_and_evaluator_view(method: str) -> None:
    jev = ScoredReranker("jev", METHODS[method], RAW["jev", method])
    kev = ScoredReranker("kev", METHODS[method], RAW["kev", method])
    for case in load_cases():
        assert jev.scores(case).keys() == kev.scores(case).keys()  # same children, same refs


def test_request_matches_kev_api_schema_and_round_trips() -> None:
    case = load_cases()[0]
    method = METHODS["score_query_in_question"]
    batch = batches(case.input, 25, 24_000)[0]
    req = request(method, case.input.query, batch, "kev-latest")
    body = req.body()
    assert set(body) == KEV_REQUEST_FIELDS
    assert {q["type"] for q in body["questions"].values()} <= KEV_QUESTION_TYPES
    assert list(body["questions"]) == [f"i{n}" for n in range(len(batch))]
    assert "query" not in body["state"]  # this method carries the query in the question
    noul = request(METHODS["noul_query_in_state"], case.input.query, batch, "jev").body()
    assert noul["state"]["query"] == case.input.query

    # A response with exactly the request's question ids pairs back onto the batch's children.
    resp = SystemOneResponse(
        model="any",
        usage=Usage(input_tokens=1, output_tokens=1),
        answers={
            k: ScoreAnswer(
                type="score", score=1.5, confidence=0.5, probabilities={"0": 0.5, "3": 0.5}
            )
            for k in body["questions"]
        },
    )
    rec = record(method, case.id, 0, batch, resp, 0.1)
    assert [(s.parent_index, s.child_index, s.question) for s in rec.scores] == [
        (it.parent_index, it.child_index, f"i{n}") for n, it in enumerate(batch)
    ]
    with pytest.raises(ValueError, match="no answer for"):
        record(method, case.id, 0, batch, resp.model_copy(update={"answers": {}}), 0.1)


def test_choice_over_children_is_one_question_with_a_child_per_option() -> None:
    """The CLM shape: state = query, options = children, score = P(option) (monotone in cosine)."""
    case = load_cases()[0]
    method = METHODS["choice_over_children"]
    (batch,) = batches(case.input, None, None)
    body = request(method, case.input.query, batch, "clm-latest").body()
    assert set(body) == KEV_REQUEST_FIELDS and body["state"] == {"query": case.input.query}
    (q,) = body["questions"].values()
    assert q["type"] == "choice" and list(q["criteria"]) == [f"i{n}" for n in range(len(batch))]
    assert q["criteria"]["i0"] == batch[0].text

    probs = {f"i{n}": 1 / len(batch) for n in range(len(batch))}
    resp = SystemOneResponse(
        model="clm-latest",
        usage=Usage(input_tokens=1, output_tokens=0),
        answers={
            CHOICE_QUESTION_ID: ChoiceAnswer(
                type="choice", choice="i0", confidence=0.0, probabilities=probs
            )
        },
    )
    rec = record(method, case.id, 0, batch, resp, 0.1)
    assert list(rec.answers) == [CHOICE_QUESTION_ID]
    assert [(s.question, s.option) for s in rec.scores] == [
        (CHOICE_QUESTION_ID, f"i{n}") for n in range(len(batch))
    ]
    s = rec.scores[3]
    assert (
        method.score(rec.answers[s.question], Binding(question=s.question, option=s.option))
        == (probs["i3"])
    )


def test_contract_rejects_mismatched_or_malformed_answers() -> None:
    with pytest.raises(ValidationError):
        RawFile.model_validate(
            {
                "run": "x",
                "written_at": "t",
                "records": [
                    {
                        "case_id": "c",
                        "batch": 0,
                        "model": "m",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                        "latency_s": 0.1,
                        "answers": {"i0": {"type": "noul"}},
                        "scores": [{"parent_index": 0, "child_index": 0, "question": "i0"}],
                    }
                ],
            }
        )
    with pytest.raises(ValidationError):  # unknown answer type: only noul / choice / score exist
        SystemOneResponse.model_validate(
            json.loads(
                '{"model":"clm","usage":{"input_tokens":1,"output_tokens":1},'
                '"answers":{"i0":{"type":"embedding","noul":0.5}}}'
            )
        )
    with pytest.raises(TypeError):  # the score method cannot score a noul answer
        METHODS["score_query_in_question"].score(
            NoulAnswer(type="noul", noul=0.5), Binding(question="i0")
        )


def test_shards_partition_cases_and_yaml_rows_share_one_experiment_id() -> None:
    """Strided shards cover every case once; every artifact of a config carries the same
    <stamp>_<petname> id, whatever the source kind (R-3)."""
    exp = load_config(REPO / "configs/kev_sizes.yaml")
    assert {s.source for s in exp.rerankers.values()} == {"production", "answers", "kev"}
    id = new_id()
    assert id.count("_") == 6 and id.split("_")[-1].replace("-", "").isalpha()
    paths = Paths(id)
    for name, src in exp.rerankers.items():
        assert paths.raw(name).name == f"raw_{name}_{id}.json.gz"
        if src.source == "kev":
            run = exp.scoring_run(id, name, src)
            seen = [c.id for s in range(run.shards) for c in run.shard_cases(s)]
            assert run.shards == 10 and sorted(seen) == sorted(c.id for c in load_cases())
    assert paths.config.name == f"config_{id}.yaml" and paths.report_base.name.endswith(id)


# laya 0.3.5 `Agent.predict` on a one-child noul request of case 0 (CPU, pinned checkpoint).
LAYA_NOUL = {
    "model": "laya-rl-agent",
    "answers": {
        "i0": {
            "type": "noul",
            "noul": 0.7752,
            "confidence": 0.7752,
            "action": {"act_probability": 1.0},
        }
    },
    "usage": {"input_tokens": 333, "output_tokens": 0},
}


def test_laya_answer_validates_and_records_like_jev() -> None:
    """Laya's extra fields (confidence, action) drop at validation; the child binds to its answer."""
    case = load_cases()[0]
    batch = batches(case.input, 1, None)[0]
    method = METHODS["noul_query_in_state"]
    resp = SystemOneResponse.model_validate(LAYA_NOUL)
    rec = record(method, case.id, 0, batch, resp, 0.03)
    assert rec.answers == {"i0": NoulAnswer(type="noul", noul=0.7752)}
    assert method.score(rec.answers["i0"], Binding(question="i0")) == 0.7752


def test_laya_source_is_one_child_per_request() -> None:
    """Laya reads 512 tokens per question, state included: more children would be cut off."""
    exp = load_config(REPO / "configs/laya_vs_jev.yaml")
    runs = [exp.scoring_run("x", n, s) for n, s in exp.rerankers.items() if s.source == "laya"]
    assert [(r.engine, r.max_items, r.concurrency, r.gpu_type) for r in runs] == [
        ("laya", 1, 1, "L4")
    ] * 2
    for bad in ({"max_items": 25}, {"method": "choice_over_children"}, {"concurrency": 8}):
        with pytest.raises(ValidationError):
            LayaSource.model_validate({"source": "laya", "method": "noul_query_in_state", **bad})


def test_noul_query_in_question_moves_the_query_into_the_question() -> None:
    case = load_cases()[0]
    batch = batches(case.input, 1, None)[0]
    plain = request(METHODS["noul_query_in_question"], case.input.query, batch, "laya").body()
    opts = request(
        METHODS["noul_query_in_question_options"], case.input.query, batch, "laya"
    ).body()
    assert plain["state"] == opts["state"] == {"items": {"i0": batch[0].text}}
    assert plain["questions"] == {
        "i0": {
            "type": "noul",
            "instructions": "Does `items.i0` contain information that answers this search: "
            f"{case.input.query}?",
        }
    }
    assert opts["questions"]["i0"]["criteria"] == {
        "true": "the passage helps answer the search",
        "false": "the passage does not help answer the search",
    }
    old = request(METHODS["noul_query_in_state"], case.input.query, batch, "jev").body()
    assert "criteria" not in old["questions"]["i0"]  # Jev's committed request is unchanged

"""Laya over many System One bodies in one forward pass (laya 0.3.5; imported in the Laya image).

`Agent.system_one` builds one sequence per question but takes a single state, so one-child
requests run as batches of 1. This builds the same sequences for many bodies, pads them into one
batch (the encoder and head mask the padding), and applies `system_one`'s temperatures unchanged."""

import numpy as np
import torch
from laya import Agent
from laya.common import (
    QTYPES,
    build_sequence,
    collate_items,
    confidence_from_probs,
    render_options,
    temp_bucket,
)

MODEL_INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


def _answer(q: dict, p: np.ndarray) -> dict:
    """`system_one`'s answer shape for the noul and score types (the ones the harness asks)."""
    if q["t"] == "noul":
        return {"type": "noul", "noul": round(float(p[1]), 4)}
    if q["t"] == "score":
        return {
            "type": "score",
            "score": round(float((np.arange(len(p)) * p).sum()), 4),
            "confidence": round(confidence_from_probs(p, len(p)), 4),
            "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
        }
    raise ValueError(f"laya batch scores noul and score questions, got type {q['t']!r}")


@torch.no_grad()
def predict_batch(agent: Agent, bodies: list[dict]) -> list[dict]:
    """One /v1/systemone response per body, every question of every body a row of one pass."""
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    rows = []
    for i, body in enumerate(bodies):
        for qid, qdef in body["questions"].items():
            q = agent._to_internal(qdef)
            seq, markers = build_sequence(agent.tok, body["state"], q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                raise ValueError(f"question {qid!r} options exceed head_max_len={head_max_len}")
            rows.append((i, qid, q, {"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]}))
    b = collate_items([[r[3] for r in rows]], agent.tok.pad_token_id)
    with torch.autocast(
        device_type=agent.device.type, dtype=agent.dtype, enabled=agent.device.type == "cuda"
    ):
        logits, _ = agent.model(*(b[k].to(agent.device) for k in MODEL_INPUTS))
    logits = logits.float().cpu().numpy()
    out = [
        {"model": "laya-rl-agent", "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
        for _ in bodies
    ]
    for r, (i, qid, q, item) in enumerate(rows):
        k, qt = len(item["markers"]), QTYPES[q["t"]]
        z = logits[r, :k] / agent.temperature_by_options.get(
            temp_bucket(qt, k), agent.temperature[qt]
        )
        p = np.exp(z - z.max())
        out[i]["answers"][qid] = _answer(q, p / p.sum())
        out[i]["usage"]["input_tokens"] += len(item["ids"])
    return out

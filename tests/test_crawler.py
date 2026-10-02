"""R-8: `python -m crawler` finds new Jev mentions and dedupes against crawler/seen.jsonl.

Offline: each source's parser gets a 1-2 item fixture recorded from the live endpoint on
2026-10-02 (tests/fixtures/crawler/); no network.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from crawler import arxiv, github, huggingface, web
from crawler.contract import (
    DEFAULT_WINDOW,
    Candidate,
    Seen,
    append_seen,
    default_since,
    new_candidates,
    read_seen,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "crawler"
SINCE = datetime(2026, 9, 25, tzinfo=UTC)
NOW = datetime(2026, 10, 2, 0, 40, tzinfo=UTC)


def test_R8_github_parse():
    payload = json.loads((FIXTURES / "github.json").read_text())
    got = github.parse(payload, "jev", SINCE, NOW)
    assert len(got) == 2
    assert got[0] == Candidate(
        source="github",
        url="https://github.com/flaviomartil/herdr-jev",
        title="flaviomartil/herdr-jev",
        snippet="Jev-driven multi-model triage and Triad orchestration plugin for Herdr and AI-Harness",
        first_seen=NOW,
        query="jev",
    )
    assert github.parse(payload, "jev", NOW + timedelta(days=1), NOW) == []


def test_R8_huggingface_parse():
    payload = json.loads((FIXTURES / "huggingface.json").read_text())
    got = huggingface.parse(payload, "kev", SINCE, NOW)
    assert [c.url for c in got] == [
        "https://huggingface.co/kevinlu4588/llmfold-aicr",
        "https://huggingface.co/ggml-org/Kev-4B-GGUF",
    ]
    assert got[1].title == "ggml-org/Kev-4B-GGUF"
    assert "decision-model" in got[1].snippet
    assert got[1].first_seen == NOW and got[1].query == "kev"


def test_R8_arxiv_parse():
    got = arxiv.parse((FIXTURES / "arxiv.xml").read_text(), '"decision model" reranker', SINCE, NOW)
    assert len(got) == 2
    assert got[0].url == "http://arxiv.org/abs/2609.37832v1"
    assert got[0].title == "Can a Cacheable Decision Model Follow Rules?"
    assert got[0].snippet.startswith("Certo is a small non-generative decision model")
    assert "\n" not in got[0].snippet
    assert arxiv.arxiv_query('"decision model" reranker') == 'all:"decision model" AND all:reranker'


def test_R8_web_parse():
    got = web.parse((FIXTURES / "web.html").read_text(), "jaredpalmer/kev", SINCE, NOW)
    assert [c.url for c in got] == [
        "https://github.com/jaredpalmer/kev",
        "https://huggingface.co/collections/jaredpalmer/kev",
    ]
    assert (
        got[0].title == "GitHub - jaredpalmer/kev: Jev-like family of decision models built on ..."
    )
    assert got[0].snippet.endswith("you can train and run on your own - jaredpalmer/kev")
    assert got[0].source == "web" and got[0].query == "jaredpalmer/kev"


@pytest.mark.parametrize("status", sorted(web.BLOCKED_STATUSES))
def test_R8_web_bot_wall_returns_empty(monkeypatch: pytest.MonkeyPatch, status: int):
    def blocked(*args, **kwargs):
        return httpx.Response(status, request=httpx.Request("GET", web.SEARCH_URL))

    monkeypatch.setattr(httpx, "get", blocked)
    assert web.search("jev", SINCE) == []


def test_R8_dedupe_against_seen(tmp_path: Path):
    def cand(url: str) -> Candidate:
        return Candidate(source="web", url=url, title="t", snippet="s", first_seen=NOW, query="jev")

    seen_path = tmp_path / "seen.jsonl"
    append_seen(seen_path, [cand("https://a")])
    seen = read_seen(seen_path)
    assert seen == [Seen(url="https://a", first_seen=NOW)]

    fresh = new_candidates([cand("https://a"), cand("https://b"), cand("https://b")], seen)
    assert [c.url for c in fresh] == ["https://b"]

    append_seen(seen_path, fresh)
    assert [s.url for s in read_seen(seen_path)] == ["https://a", "https://b"]


def test_R8_since_defaults_to_last_run_else_7_days():
    assert default_since([], NOW) == NOW - DEFAULT_WINDOW
    seen = [
        Seen(url="https://a", first_seen=NOW - timedelta(days=3)),
        Seen(url="https://b", first_seen=NOW - timedelta(days=1)),
    ]
    assert default_since(seen, NOW) == NOW - timedelta(days=1)

"""R-8: `python -m crawler` finds new Jev mentions and dedupes against crawler/seen.jsonl.

Offline: each source's parser gets a 1-2 item fixture recorded from the live endpoint on
2026-10-02 (tests/fixtures/crawler/); no network.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from crawler import arxiv, github, hackernews, huggingface, submitted, twitter, web
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
        key="https://github.com/flaviomartil/herdr-jev",
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
    assert got[1].key == "hf:ggml-org/Kev-4B-GGUF@d924f2e2c3872da8b8aaf3eb4453b4126deceb79"
    assert "decision-model" in got[1].snippet
    assert got[1].first_seen == NOW and got[1].query == "kev"


def test_R8_arxiv_parse():
    got = arxiv.parse((FIXTURES / "arxiv.xml").read_text(), '"decision model" reranker', SINCE, NOW)
    assert len(got) == 2
    assert got[0].url == "http://arxiv.org/abs/2609.37832v1" and got[0].key == got[0].url
    assert got[0].title == "Can a Cacheable Decision Model Follow Rules?"
    assert got[0].snippet.startswith("Certo is a small non-generative decision model")
    assert "\n" not in got[0].snippet
    assert arxiv.arxiv_query('"decision model" reranker') == 'all:"decision model" AND all:reranker'


def test_R8_web_parse():
    payload = json.loads((FIXTURES / "web.json").read_text())
    got = web.parse(payload, "jaredpalmer/kev", SINCE, NOW)
    assert [c.url for c in got] == [
        "https://x.com/jaredpalmer/status/2101715352258232539",
        "https://github.com/jaredpalmer/kev",
    ]
    assert got[1].key == got[1].url
    assert got[1].title.startswith(
        "GitHub - jaredpalmer/kev: tiny Jev-like family of decision models"
    )
    assert got[1].snippet.startswith("``` uv run python -m kev.train --data train.jsonl")
    assert "\n" not in got[0].snippet
    assert got[0].source == "web" and got[0].query == "jaredpalmer/kev"
    assert web.days_back(SINCE, NOW) == 8 and web.days_back(NOW, NOW) == web.MIN_DAYS


@pytest.mark.parametrize("status", sorted(web.BLOCKED_STATUSES))
def test_R8_web_bot_wall_returns_empty(monkeypatch: pytest.MonkeyPatch, status: int):
    def blocked(*args, **kwargs):
        return httpx.Response(status, request=httpx.Request("POST", web.SEARCH_URL))

    monkeypatch.setattr(httpx, "post", blocked)
    assert web.search("jev", SINCE) == []
    assert twitter.search("jev", SINCE) == []


def test_R8_twitter_parse_keeps_only_x_hosts():
    payload = json.loads((FIXTURES / "twitter.json").read_text())
    got = twitter.parse(payload, "jaredpalmer kev", SINCE, NOW)
    assert [c.url for c in got] == [
        "https://x.com/jaredpalmer/highlights",
        "https://x.com/jaredpalmer/status/2101715352258232539",
        "https://x.com/OpenRouter/status/2103560670205886744",
    ]
    assert got[1].title.startswith('Jared Palmer on X: "UPDATE: Kev-0.6B, 4B, and 8B are now ava')
    assert got[1].source == "twitter" and got[1].key == got[1].url
    assert got[1].query == "jaredpalmer kev"

    off_site = {"url": "https://github.com/jaredpalmer/kev", "title": "t", "content": "c"}
    mixed = {"results": [off_site, *payload["results"][:1]]}
    assert [c.url for c in twitter.parse(mixed, "kev", SINCE, NOW)] == [got[0].url]


def test_R8_twitter_search_appends_site_filter(monkeypatch: pytest.MonkeyPatch):
    sent: dict = {}

    def capture(url: str, headers: dict, json: dict, timeout: float) -> httpx.Response:
        sent.update(json)
        return httpx.Response(200, json={"results": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", capture)
    assert twitter.search("jaredpalmer kev", SINCE) == []
    assert sent["query"] == "jaredpalmer kev" + twitter.SITE_FILTER
    assert sent["max_results"] == web.MAX_RESULTS and sent["days"] >= web.MIN_DAYS


def test_R8_hackernews_parse():
    payload = json.loads((FIXTURES / "hackernews.json").read_text())
    got = hackernews.parse(payload, '"jaredpalmer"', datetime(2026, 9, 20, tzinfo=UTC), NOW)
    assert [c.url for c in got] == [
        "https://news.ycombinator.com/item?id=49928555",
        "https://news.ycombinator.com/item?id=49783999",
    ]
    comment, story = got
    assert comment.title == "OpenAI has a LOT of work to do if they think Luna can compete with Jev"
    assert comment.snippet.startswith(
        "apples and oranges, also Clef and Kev (the middle path) https://blog.cloudflare.com/"
    )
    assert "<" not in comment.snippet and len(comment.snippet) <= hackernews.SNIPPET_CHARS
    assert story.title == "Kev: Tiny Jev-like family of decision models built on top of Qwen3.5"
    assert story.snippet == "https://github.com/jaredpalmer/kev/tree/main"
    assert (
        story.source == "hackernews" and story.key == story.url and story.query == '"jaredpalmer"'
    )
    assert [c.url for c in hackernews.parse(payload, '"jaredpalmer"', SINCE, NOW)] == [comment.url]


def cand(url: str, key: str | None = None) -> Candidate:
    return Candidate(
        source="web", url=url, key=key or url, title="t", snippet="s", first_seen=NOW, query="jev"
    )


def test_R8_dedupe_against_seen(tmp_path: Path):
    seen_path = tmp_path / "seen.jsonl"
    append_seen(seen_path, [cand("https://a")])
    seen = read_seen(seen_path)
    assert seen == [Seen(url="https://a", key="https://a", first_seen=NOW)]

    fresh = new_candidates([cand("https://a"), cand("https://b"), cand("https://b")], seen)
    assert [c.url for c in fresh] == ["https://b"]

    append_seen(seen_path, fresh)
    assert [s.key for s in read_seen(seen_path)] == ["https://a", "https://b"]


def test_R8_dedupe_is_on_key_so_new_hf_sha_resurfaces(tmp_path: Path):
    url = "https://huggingface.co/jaredpalmer/kev-27b"
    v1 = cand(url, "hf:jaredpalmer/kev-27b@01b8199")
    v2 = cand(url, "hf:jaredpalmer/kev-27b@f00dcafe")
    assert new_candidates([v1, v2], []) == [v1, v2]
    seen_path = tmp_path / "seen.jsonl"
    append_seen(seen_path, [v1])
    assert new_candidates([v1, v2], read_seen(seen_path)) == [v2]


def test_R8_read_seen_rejects_line_without_key(tmp_path: Path):
    seen_path = tmp_path / "seen.jsonl"
    seen_path.write_text('{"url":"https://a","first_seen":"2026-10-02T00:40:00Z"}\n')
    with pytest.raises(ValidationError):
        read_seen(seen_path)


def test_R8_since_defaults_to_last_run_else_7_days():
    assert default_since([], NOW) == NOW - DEFAULT_WINDOW
    seen = [
        Seen(url="https://a", key="https://a", first_seen=NOW - timedelta(days=3)),
        Seen(url="https://b", key="https://b", first_seen=NOW - timedelta(days=1)),
    ]
    assert default_since(seen, NOW) == NOW - timedelta(days=1)


def test_R8_submitted_requests_become_candidates(tmp_path: Path):
    day = tmp_path / "requests" / "2026-10-02"
    day.mkdir(parents=True)
    (day / "070509_hf.json").write_text(
        '{"url": "https://huggingface.co/org/model", "submitted_at": "2026-10-02T07:05:09Z"}\n'
    )
    (tmp_path / "requests" / "README.md").write_text("ignored\n")
    [c] = submitted.load(tmp_path / "requests")
    assert (c.source, c.key, c.url) == ("submitted", "https://huggingface.co/org/model", c.url)
    assert c.first_seen == datetime(2026, 10, 2, 7, 5, 9, tzinfo=UTC)
    assert submitted.load(tmp_path / "nowhere") == []

"""R-9: open `evaluate` issues become configs/issue_<number>_<slug>.yaml (offline, fixture payload)."""

import json
from pathlib import Path

import pytest
import yaml

from jev_tracker.evaluate_issues import (
    Issue,
    config_path,
    config_text,
    parse_issue,
    parse_issues,
    slug,
    write_config,
)
from jev_tracker.experiment import CONFIGS_DIR, Experiment, KevSource

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "evaluate_issues.json"
ISSUES = parse_issues(json.loads(FIXTURE.read_text()))


def test_R9_parse_issue_yields_first_yaml_block():
    assert len(ISSUES) == 2 and ISSUES[0].number == 7
    yaml_text = parse_issue(ISSUES[0].body)
    assert yaml_text.startswith("name: kev7b\n") and yaml_text.endswith(
        "method: noul_query_in_state\n"
    )
    assert parse_issue("a\n```yaml\nname: x\n```\nb\n```yaml\nname: y\n```") == "name: x\n"
    with pytest.raises(ValueError, match="no ```yaml block"):
        parse_issue("no config here")


def test_R9_config_text_validates_as_experiment():
    text = config_text(ISSUES[0])
    assert text.startswith(
        "# https://github.com/marcus-rox/jev-tracker/issues/7: evaluate: kev7b\n"
    )
    experiment = Experiment.model_validate(yaml.safe_load(text))
    assert experiment.name == "kev7b"
    assert experiment.rerankers["kev7b_noul"] == KevSource(
        source="kev",
        model="jaredpalmer/kev-7b",
        max_items=12,
        max_chars=12000,
        shards=3,
        concurrency=16,
        method="noul_query_in_state",
    )


def test_R9_missing_yaml_block_error_names_the_issue():
    with pytest.raises(ValueError, match=r"issue #8 'evaluate: Laya Typed': no ```yaml block"):
        config_text(ISSUES[1])


def test_R9_name_derived_from_title_when_missing():
    issue = Issue(
        number=9,
        title="evaluate: Kev 27B v2",
        body="```yaml\nrerankers:\n  prod:\n    source: production\n```",
    )
    assert slug(issue.title) == "kev_27b_v2"
    assert config_text(issue).splitlines()[1] == "name: kev_27b_v2"
    assert config_path(9, issue.title) == CONFIGS_DIR / "issue_9_kev_27b_v2.yaml"
    with pytest.raises(ValueError, match=r"(?s)issue #9 .*rerankers.*Field required"):
        config_text(Issue(number=9, title=issue.title, body="```yaml\nname: x\n```"))


def test_R9_write_config_skips_existing(tmp_path: Path):
    first = write_config(ISSUES[0], tmp_path)
    path = tmp_path / "issue_7_kev7b.yaml"
    assert first == f"#7 'evaluate: kev7b': wrote {path}" and path.exists()
    written = path.read_text()
    assert write_config(ISSUES[0], tmp_path) == f"#7 'evaluate: kev7b': skipped, {path} exists"
    assert path.read_text() == written

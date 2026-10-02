"""R-9: open GitHub issues labelled `evaluate` become configs/issue_<number>_<slug>.yaml.

    uv run python -m jev_tracker.evaluate_issues            # GITHUB_TOKEN optional (rate limit)

The site's "Evaluate a new model" form (site/src/evaluate_config.ts) opens an issue titled
`evaluate: <name>` whose body holds the experiment config in a ```yaml block. This module lists
those issues, validates each block as an `Experiment` and writes it under configs/, skipping files
that already exist so a rerun never overwrites a config the automation already ran.
"""

import os
import re
import sys
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel, ValidationError

from jev_tracker.experiment import CONFIGS_DIR, Experiment

ISSUES_URL = "https://api.github.com/repos/marcus-rox/jev-tracker/issues"
ISSUE_HTML_URL = "https://github.com/marcus-rox/jev-tracker/issues"
LABEL = "evaluate"
TOKEN_ENV = "GITHUB_TOKEN"
PER_PAGE = 100
HTTP_TIMEOUT_SECONDS = 30.0
TITLE_PREFIX = "evaluate:"
YAML_BLOCK = re.compile(r"```ya?ml[ \t]*\n(.*?)```", re.DOTALL)
NOT_SLUG = re.compile(r"[^a-z0-9]+")


class Issue(BaseModel):
    model_config = {"frozen": True}

    number: int
    title: str
    body: str | None = None


def parse_issues(payload: list[dict]) -> list[Issue]:
    """Issues from the API payload; pull requests (which the issues endpoint also lists) are dropped."""
    return [Issue.model_validate(item) for item in payload if "pull_request" not in item]


def fetch_issues() -> list[Issue]:
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get(TOKEN_ENV):
        headers["Authorization"] = f"Bearer {token}"
    params = {"labels": LABEL, "state": "open", "per_page": PER_PAGE}
    response = httpx.get(ISSUES_URL, params=params, headers=headers, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return parse_issues(response.json())


def parse_issue(body: str) -> str:
    """The text of the first ```yaml block in an issue body."""
    match = YAML_BLOCK.search(body)
    if match is None:
        raise ValueError(f"no ```yaml block in issue body {body!r}")
    return match.group(1)


def slug(title: str) -> str:
    stripped = title.strip()
    if stripped.lower().startswith(TITLE_PREFIX):
        stripped = stripped[len(TITLE_PREFIX) :]
    cleaned = NOT_SLUG.sub("_", stripped.strip().lower()).strip("_")
    if not cleaned:
        raise ValueError(f"issue title {title!r} leaves no slug")
    return cleaned


def config_path(issue_number: int, title: str, configs_dir: Path = CONFIGS_DIR) -> Path:
    return configs_dir / f"issue_{issue_number}_{slug(title)}.yaml"


def config_text(issue: Issue) -> str:
    """The issue's yaml block, validated as an Experiment, with the issue url as its header comment."""
    try:
        yaml_text = parse_issue(issue.body or "")
        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            raise ValueError(f"yaml block is {type(data).__name__}, expected a mapping: {data!r}")
        if "name" not in data:
            data = {"name": slug(issue.title), **data}
            yaml_text = f"name: {data['name']}\n{yaml_text}"
        Experiment.model_validate(data)
    except (ValueError, ValidationError, yaml.YAMLError) as error:
        raise ValueError(f"issue #{issue.number} {issue.title!r}: {error}") from error
    header = f"# {ISSUE_HTML_URL}/{issue.number}: {issue.title}\n"
    return header + yaml_text.rstrip("\n") + "\n"


def write_config(issue: Issue, configs_dir: Path = CONFIGS_DIR) -> str:
    """Write the issue's config unless it already exists; the printed line for that issue."""
    text = config_text(issue)
    path = config_path(issue.number, issue.title, configs_dir)
    if path.exists():
        return f"#{issue.number} {issue.title!r}: skipped, {path} exists"
    path.write_text(text)
    return f"#{issue.number} {issue.title!r}: wrote {path}"


def main() -> int:
    issues = fetch_issues()
    if not issues:
        print(f"0 {LABEL} issues")
        return 0
    failed = 0
    for issue in issues:
        try:
            print(write_config(issue))
        except ValueError as error:  # boundary: one malformed issue must not stop the others
            failed += 1
            print(f"FAILED {error}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

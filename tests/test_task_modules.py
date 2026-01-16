"""Smoke tests for split task modules."""
from __future__ import annotations

from pathlib import Path

from src.tasks.anti_patterns import detect_anti_patterns
from src.tasks.duplication import detect_duplicate_blocks
from src.tasks.health import health_score
from src.tasks.security import security_heuristics


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_security_heuristics_detects_eval(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "app.py", "def run():\n    eval('1+1')\n")

    result = security_heuristics(source_path=str(repo))

    assert result["counts"]["total"] >= 1
    by_rule = result["counts"].get("by_rule", {})
    assert "py-ast-eval-exec" in by_rule or "py-eval-exec" in by_rule


def test_detect_anti_patterns_flags_missing_tests(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "app.py", "def hello():\n    return 'ok'\n")

    result = detect_anti_patterns(source_path=str(repo))

    issue_types = {issue.get("type") for issue in result.get("issues", [])}
    assert "missing_tests" in issue_types


def test_health_score_returns_metrics(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "README.md", "# Test Repo")
    _write(repo / "app.py", "def hello():\n    return 'ok'\n")

    result = health_score(source_path=str(repo))

    assert "score" in result
    assert "details" in result
    assert "modularity" in result["details"]


def test_detect_duplicate_blocks_basic(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    snippet = """def foo():
    x = 1
    y = 2
    return x + y
"""
    _write(repo / "a.py", snippet)
    _write(repo / "b.py", snippet)

    result = detect_duplicate_blocks(source_path=str(repo), min_lines=3)

    assert result["scanned_files"] >= 2
    assert result["count"] >= 1

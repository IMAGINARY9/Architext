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


def test_security_heuristics_respects_max_findings(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(
        repo / "danger.py",
        "def run(x):\n"
        "    eval('1+1')\n"
        "    eval('2+2')\n"
        "    return open(x).read()\n",
    )

    result = security_heuristics(source_path=str(repo), max_findings=1)

    assert result["counts"]["total"] == 1


def test_security_heuristics_returns_deterministic_order(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "b.py", "def run():\n    eval('1+1')\n")
    _write(repo / "a.py", "def run():\n    eval('1+1')\n")

    result = security_heuristics(source_path=str(repo))
    files = [item.get("file") for item in result["findings"]]

    assert files == sorted(files)


def test_security_heuristics_zero_max_findings_returns_empty(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(repo / "a.py", "def run():\n    eval('1+1')\n")

    result = security_heuristics(source_path=str(repo), max_findings=0)

    assert result["findings"] == []
    assert result["counts"]["total"] == 0
    assert result["counts"]["by_rule"] == {}
    assert result["counts"]["by_severity"] == {}


def test_security_heuristics_detects_subprocess_shell_true(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(
        repo / "cmd.py",
        "import subprocess\n"
        "def run(cmd):\n"
        "    subprocess.run(cmd, shell=True)\n",
    )

    result = security_heuristics(source_path=str(repo))
    by_rule = result["counts"].get("by_rule", {})

    assert "py-ast-subprocess-shell-true" in by_rule


def test_security_heuristics_detects_yaml_load_without_safe_loader(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(
        repo / "parse.py",
        "import yaml\n"
        "def parse(text):\n"
        "    return yaml.load(text)\n",
    )

    result = security_heuristics(source_path=str(repo))
    by_rule = result["counts"].get("by_rule", {})

    assert "py-ast-yaml-unsafe-load" in by_rule


def test_security_heuristics_detects_taint_in_keyword_argument(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write(
        repo / "keyword_taint.py",
        "def run(user_path):\n"
        "    safe = user_path\n"
        "    return open(file=safe)\n",
    )

    result = security_heuristics(source_path=str(repo))
    rules = {finding.get("rule_id") for finding in result["findings"]}

    assert "py-ast-taint-flow" in rules

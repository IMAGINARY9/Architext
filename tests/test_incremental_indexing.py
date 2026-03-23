"""Tests for incremental indexing prototype helpers."""
from __future__ import annotations

from pathlib import Path

from src.indexer_components.incremental import (
    build_manifest,
    diff_manifests,
    select_incremental_index_targets,
)


def test_build_manifest_and_diff_detect_modifications(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    file_a = repo / "a.py"
    file_b = repo / "b.py"
    file_a.write_text("print('a')", encoding="utf-8")
    file_b.write_text("print('b')", encoding="utf-8")

    baseline = build_manifest(str(repo))

    file_b.write_text("print('changed')", encoding="utf-8")
    current = build_manifest(str(repo))

    diff = diff_manifests(baseline, current)

    assert diff.added == []
    assert diff.removed == []
    assert str(file_b) in diff.modified


def test_incremental_targets_fallback_when_change_ratio_high() -> None:
    previous = {
        "a.py": "1",
        "b.py": "1",
        "c.py": "1",
    }
    current = {
        "a.py": "2",
        "b.py": "2",
        "c.py": "1",
    }

    targets, fallback = select_incremental_index_targets(
        previous,
        current,
        change_ratio_threshold=0.5,
    )

    assert fallback is True
    assert targets == ["a.py", "b.py", "c.py"]


def test_incremental_targets_no_fallback_when_changes_small() -> None:
    previous = {
        "a.py": "1",
        "b.py": "1",
        "c.py": "1",
        "d.py": "1",
    }
    current = {
        "a.py": "1",
        "b.py": "1",
        "c.py": "2",
        "d.py": "1",
    }

    targets, fallback = select_incremental_index_targets(
        previous,
        current,
        change_ratio_threshold=0.5,
    )

    assert fallback is False
    assert targets == ["c.py"]

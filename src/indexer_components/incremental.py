"""Incremental indexing prototype helpers.

This module computes file manifests, diffs changes between snapshots,
and provides fallback heuristics when change volume is too high.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.indexer import gather_index_files


FileManifest = Dict[str, str]


@dataclass(frozen=True)
class ManifestDiff:
    """Summary of file changes between two manifests."""

    added: List[str]
    modified: List[str]
    removed: List[str]

    @property
    def changed(self) -> List[str]:
        """Files that should be re-indexed for incremental updates."""
        return sorted(set(self.added + self.modified))


def file_fingerprint(path: str) -> str:
    """Compute a lightweight file fingerprint using size and mtime_ns."""
    stat = Path(path).stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def build_manifest(
    source_path: str,
    max_files: Optional[int] = None,
) -> FileManifest:
    """Build a manifest for indexable files in the source path."""
    files = gather_index_files(source_path, max_files=max_files)
    return {path: file_fingerprint(path) for path in files}


def diff_manifests(previous: FileManifest, current: FileManifest) -> ManifestDiff:
    """Compute added/modified/removed files between two manifests."""
    previous_paths = set(previous)
    current_paths = set(current)

    added = sorted(current_paths - previous_paths)
    removed = sorted(previous_paths - current_paths)
    modified = sorted(
        path for path in current_paths & previous_paths if previous[path] != current[path]
    )

    return ManifestDiff(added=added, modified=modified, removed=removed)


def should_fallback_to_full_scan(
    total_files: int,
    changed_files: int,
    change_ratio_threshold: float = 0.35,
) -> bool:
    """Return True when incremental update is likely less efficient than full scan."""
    if total_files <= 0:
        return False
    ratio = changed_files / total_files
    return ratio >= change_ratio_threshold


def select_incremental_index_targets(
    previous: FileManifest,
    current: FileManifest,
    change_ratio_threshold: float = 0.35,
) -> Tuple[List[str], bool]:
    """Select incremental targets and whether to fallback to full indexing."""
    diff = diff_manifests(previous, current)
    changed = diff.changed
    fallback_to_full = should_fallback_to_full_scan(
        total_files=max(len(current), 1),
        changed_files=len(changed),
        change_ratio_threshold=change_ratio_threshold,
    )
    if fallback_to_full:
        return sorted(current.keys()), True
    return changed, False

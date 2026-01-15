"""Shared file filtering utilities for indexing and analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set


DEFAULT_SKIP_FRAGMENTS: Set[str] = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    ".env",
    "dist",
    "build",
    ".pytest_cache",
    "models_cache",
    "storage",
}


def should_skip_path(path: Path, additional_skip_fragments: Iterable[str] | None = None) -> bool:
    """Return True if a path should be skipped during scanning."""
    fragments = set(DEFAULT_SKIP_FRAGMENTS)
    if additional_skip_fragments:
        fragments.update(additional_skip_fragments)

    for part in path.parts:
        lower = part.lower()
        if lower == "storage" or lower.startswith("storage-") or lower.startswith("storage_"):
            return True
        if lower in fragments:
            return True
        if part.startswith("."):
            return True
    return False

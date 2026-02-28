"""Shared server utilities used by route handlers and the app factory.

Contains pure helper functions (hashing, directory stats, text transforms)
and index-resolution helpers that raise ``HTTPException`` on validation failures.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory token-bucket rate limiter keyed by client IP."""

    def __init__(self, rate_per_minute: int, burst: Optional[int] = None) -> None:
        self.rate_per_minute = max(rate_per_minute, 1)
        self.capacity = burst or self.rate_per_minute
        self.tokens: Dict[str, float] = {}
        self.last_refill: Dict[str, float] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            tokens = self.tokens.get(key, self.capacity)
            last = self.last_refill.get(key, now)
            elapsed = max(0.0, now - last)
            refill = elapsed * (self.rate_per_minute / 60.0)
            tokens = min(self.capacity, tokens + refill)
            if tokens < 1.0:
                self.tokens[key] = tokens
                self.last_refill[key] = now
                return False
            self.tokens[key] = tokens - 1.0
            self.last_refill[key] = now
            return True


# ---------------------------------------------------------------------------
# Path validation helpers
# ---------------------------------------------------------------------------

def is_within_any(candidate: Path, roots: List[Path]) -> bool:
    """Return True if *candidate* is contained within any of *roots*."""
    for root in roots:
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def parse_allowed_roots(
    raw: Optional[str], defaults: List[Path]
) -> List[Path]:
    """Parse an OS-pathsep-separated string into a list of resolved Paths."""
    if raw:
        return [
            Path(item).expanduser().resolve()
            for item in raw.split(os.pathsep)
            if item.strip()
        ]
    return defaults


# ---------------------------------------------------------------------------
# Storage / index helpers
# ---------------------------------------------------------------------------

def clear_chroma_storage(storage_path: str) -> None:
    """Clear Chroma SQLite files to avoid duplicate inserts on reindex."""
    base = Path(storage_path)
    if not base.exists():
        return
    for candidate in base.glob("chroma.sqlite3*"):
        if candidate.is_file():
            try:
                candidate.unlink()
            except OSError:
                logger.debug("Could not remove %s, skipping", candidate)
                continue


def compute_source_hash(source: str) -> str:
    """Compute a stable short hash for a source string."""
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def compute_index_name(source: str) -> str:
    """Generate a meaningful index name from the source path or URL."""
    if source.startswith(("http://", "https://", "git@", "ssh://")):
        match = re.search(r"/([^/]+?)(\.git)?$", source)
        name = match.group(1) if match else "repo"
    else:
        name = Path(source).name or "local"

    # Sanitize: keep only alnum, _, -
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    if len(name) > 20:
        name = name[:20]

    hash_part = hashlib.sha256(source.encode()).hexdigest()[:8]
    return f"{name}-{hash_part}"


# ---------------------------------------------------------------------------
# Directory stat helpers
# ---------------------------------------------------------------------------

def dir_disk_usage(path: Path) -> int:
    """Return total size in bytes for files under *path*."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except Exception:
                continue
    return total


def dir_last_modified(path: Path) -> Optional[str]:
    """Return ISO-8601 UTC timestamp of the latest file modification in *path*."""
    try:
        latest = max(p.stat().st_mtime for p in path.rglob("*"))
        return (
            datetime.fromtimestamp(latest, timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
    except Exception:
        return None


def dir_file_count(path: Path) -> int:
    """Return the number of files under *path* (recursive)."""
    count = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                count += 1
        except Exception:
            continue
    return count


# ---------------------------------------------------------------------------
# MCP tools schema builder
# ---------------------------------------------------------------------------

def build_mcp_tools_schema(storage_roots: List[Path]) -> List[Dict[str, Any]]:
    """Build MCP tools schema with available index names pre-filled."""
    indices: List[str] = []
    try:
        for root in storage_roots:
            if not root.exists():
                continue
            if (root / "chroma.sqlite3").exists():
                indices.append(root.name)
            for item in root.iterdir():
                if item.is_dir() and (item / "chroma.sqlite3").exists():
                    indices.append(item.name)
    except Exception:
        logger.debug("Error scanning storage roots for indices", exc_info=True)

    available = ", ".join(indices) if indices else "none found"
    return [
        {
            "name": "architext.query",
            "description": "Query an index and return agent/human output.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Query text to ask the index",
                    },
                    "name": {
                        "type": "string",
                        "description": (
                            f"Name of the index to query. Available indices: {available}"
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["rag", "agent"],
                        "description": (
                            "Response mode: 'rag' (retrieval-augmented generation) "
                            "for free text, 'agent' for structured output"
                        ),
                    },
                    "compact": {
                        "type": "boolean",
                        "description": (
                            "When true in agent mode, returns a compact agent schema"
                        ),
                    },
                    "enable_hybrid": {
                        "type": "boolean",
                        "description": "Enable hybrid (keyword+vector) search",
                    },
                    "hybrid_alpha": {
                        "type": "number",
                        "description": "Blend factor for hybrid scoring",
                    },
                    "enable_rerank": {
                        "type": "boolean",
                        "description": "Enable re-ranking of retrieved results",
                    },
                    "rerank_model": {
                        "type": "string",
                        "description": "Model to use for reranking",
                    },
                    "rerank_top_n": {
                        "type": "integer",
                        "description": "How many top candidates to rerank",
                    },
                },
                "required": ["text"],
            },
        },
        {
            "name": "architext.task",
            "description": "Run an analysis task synchronously.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": (
                            "Task name (e.g., 'analyze-structure', 'tech-stack', "
                            "'detect-anti-patterns')"
                        ),
                    },
                    "storage": {
                        "type": "string",
                        "description": (
                            "Optional index storage path to analyze. "
                            f"Available indices: {available}"
                        ),
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Source repository path or URL to run the task against"
                        ),
                    },
                    "output_format": {
                        "type": "string",
                        "description": "Output format for the task result",
                    },
                    "depth": {
                        "type": "string",
                        "description": "Analysis depth level",
                    },
                    "module": {
                        "type": "string",
                        "description": "Module name for module-specific tasks",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": (
                            "Directory to write outputs for tasks that produce files"
                        ),
                    },
                },
                "required": ["task"],
            },
        },
        {
            "name": "architext.list_indices",
            "description": "List all available indices in storage.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "architext.get_index_metadata",
            "description": "Get detailed metadata for a specific index.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "index_name": {
                        "type": "string",
                        "description": (
                            f"Name of the index. Available indices: {available}"
                        ),
                    },
                },
                "required": ["index_name"],
            },
        },
    ]

# ---------------------------------------------------------------------------
# Timestamp / text-transform helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string (``…Z`` suffix)."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def ensure_sources_instruction(text: str) -> str:
    """Append a source-citation instruction unless the query already asks for sources."""
    lower = (text or "").lower()
    keywords = [
        "show sources", "show source", "file path", "file:",
        "quote line", "quote lines", "line", "lines", "sources", "cite",
    ]
    if any(k in lower for k in keywords):
        return text
    instruction = (
        "\n\nPlease include sources (file path and line ranges) "
        "for any code referenced, e.g., 'file.py:10-20'."
    )
    return text + instruction


# ---------------------------------------------------------------------------
# Index-resolution helpers  (raise ``HTTPException`` on validation failures)
# ---------------------------------------------------------------------------

def find_index_path_by_name(name: str, storage_roots: List[Path]) -> str:
    """Return the storage path for an index identified by *name*.

    Raises ``HTTPException(404)`` when no matching index is found.
    """
    from fastapi import HTTPException  # lazy import

    for root in storage_roots:
        if not root.exists():
            continue
        if root.name == name and (root / "chroma.sqlite3").exists():
            return str(root)
        for item in root.iterdir():
            if item.is_dir() and item.name == name and (item / "chroma.sqlite3").exists():
                return str(item)
    raise HTTPException(status_code=404, detail=f"Index with name '{name}' not found")


def list_available_indices(storage_roots: List[Path]) -> List[Dict[str, str]]:
    """Return a list of ``{"name": …, "path": …}`` dicts for every discovered index."""
    found: List[Dict[str, str]] = []
    for root in storage_roots:
        if not root.exists():
            continue
        if (root / "chroma.sqlite3").exists():
            found.append({"name": root.name, "path": str(root)})
        for item in root.iterdir():
            if item.is_dir() and (item / "chroma.sqlite3").exists():
                found.append({"name": item.name, "path": str(item)})
    return found


def resolve_index_storage(name: Optional[str], storage_roots: List[Path]) -> str:
    """Resolve the storage path for an index.

    When *name* is given, look it up directly.  Otherwise auto-select iff
    exactly one index exists.

    Raises ``HTTPException(404)`` if none found, ``HTTPException(400)`` if
    multiple exist and *name* was not specified.
    """
    from fastapi import HTTPException  # lazy import

    if name:
        return find_index_path_by_name(name, storage_roots)

    candidates = list_available_indices(storage_roots)
    if len(candidates) == 1:
        return candidates[0]["path"]
    if len(candidates) == 0:
        raise HTTPException(
            status_code=404,
            detail="No indices available. Create one with the /index endpoint.",
        )
    names = ", ".join(c["name"] for c in candidates[:10])
    raise HTTPException(
        status_code=400,
        detail=(
            f"Multiple indices available ({len(candidates)}). "
            f"Specify the 'name' field to select one. Available: {names}"
        ),
    )


def validate_source_dir(path: Path, source_roots: List[Path]) -> None:
    """Validate that *path* exists, is a directory, and lives under *source_roots*.

    Raises ``HTTPException(400)`` on any validation failure.
    """
    from fastapi import HTTPException  # lazy import

    if not path.exists():
        raise HTTPException(status_code=400, detail="source path does not exist")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail="source path must be a directory")
    if not is_within_any(path, source_roots):
        raise HTTPException(
            status_code=400, detail="source path must be within allowed roots"
        )


def resolve_storage_path(
    raw_path: Optional[str],
    source: Optional[str],
    storage_roots: List[Path],
    default_storage_path: str,
) -> str:
    """Resolve and validate the storage path for indexing.

    If *raw_path* is given it is validated against *storage_roots*.
    Otherwise a deterministic path is computed from *source* (or the default
    storage path is used as fallback).

    Raises ``HTTPException(400)`` on validation failures.
    """
    from fastapi import HTTPException  # lazy import

    if raw_path:
        candidate = Path(raw_path).expanduser().resolve()
        if not is_within_any(candidate, storage_roots):
            allowed_paths = ", ".join(str(root) for root in storage_roots)
            if raw_path in ["string", "path", "example", "null"]:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"storage path cannot be '{raw_path}' (placeholder value). "
                        f"Use a real path within allowed roots like "
                        f"'{allowed_paths}/my-index', or omit the field to use defaults."
                    ),
                )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"storage path '{candidate}' must be within "
                    f"allowed roots: {allowed_paths}"
                ),
            )
        return str(candidate)

    if source:
        index_name = compute_index_name(source)
        default_root = Path(default_storage_path).expanduser().resolve()
        storage_path = default_root / index_name
        if not is_within_any(storage_path, storage_roots):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Computed storage path '{storage_path}' is not within allowed roots"
                ),
            )
        return str(storage_path)

    candidate = Path(default_storage_path).expanduser().resolve()
    if not is_within_any(candidate, storage_roots):
        raise HTTPException(
            status_code=400,
            detail=f"Default storage path '{candidate}' is not within allowed roots",
        )
    return str(candidate)
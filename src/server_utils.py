"""Shared server utilities used by route handlers and the app factory.

Pure helper functions with no FastAPI dependency — safe to import
from any module that needs path validation, hashing, or directory stats.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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
            except Exception:
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
        pass

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

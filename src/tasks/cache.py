"""Compatibility shim — re-exports from src.tasks.core.cache.

All caching implementations live in src/tasks/core/cache.py.
Import from ``src.tasks.core`` or ``src.tasks`` for the public API.
"""
from __future__ import annotations

from src.tasks.core.cache import (  # noqa: F401 — re-exports
    CacheEntry,
    TaskResultCache,
    cached_task,
    get_task_cache,
)

__all__ = [
    "CacheEntry",
    "TaskResultCache",
    "cached_task",
    "get_task_cache",
]

"""Task execution context — backward-compatibility re-export shim.

Canonical implementation lives in ``src.tasks.shared``.
This module re-exports the public API so that existing imports from
``src.tasks.core.context`` continue to work transparently.
"""
from src.tasks.shared import (  # noqa: F401
    TaskContext,
    get_current_context,
    set_current_context,
    task_context,
)

__all__ = [
    "TaskContext",
    "get_current_context",
    "set_current_context",
    "task_context",
]

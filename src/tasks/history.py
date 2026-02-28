"""Compatibility shim — re-exports from src.tasks.orchestration.history.

All implementations live in src/tasks/orchestration/history.py.
Import from ``src.tasks.orchestration`` or ``src.tasks`` for the public API.
"""
from src.tasks.orchestration.history import *  # noqa: F401,F403
from src.tasks.orchestration.history import (  # explicit re-exports for type checkers
    TaskExecution,
    TaskAnalytics,
    TaskExecutionHistory,
    ExecutionTracker,
    get_task_history,
)

__all__ = [
    "TaskExecution",
    "TaskAnalytics",
    "TaskExecutionHistory",
    "ExecutionTracker",
    "get_task_history",
]

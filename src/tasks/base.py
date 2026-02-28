"""Compatibility shim — re-exports from src.tasks.core.base.

All implementations live in src/tasks/core/base.py.
Import from ``src.tasks.core`` or ``src.tasks`` for the public API.
"""
from __future__ import annotations

from src.tasks.core.base import (  # noqa: F401 — re-exports
    BaseTask,
    FileInfo,
    TaskResult,
    PYTHON_EXTENSIONS,
    JAVASCRIPT_EXTENSIONS,
    TYPESCRIPT_EXTENSIONS,
    JS_TS_EXTENSIONS,
    CODE_EXTENSIONS,
    DOCUMENTATION_EXTENSIONS,
    CONFIG_EXTENSIONS,
    filter_files_by_extension,
    count_by_extension,
    count_by_language,
    get_test_files,
    get_documentation_files,
    calculate_ratio,
)

__all__ = [
    "BaseTask",
    "FileInfo",
    "TaskResult",
    "PYTHON_EXTENSIONS",
    "JAVASCRIPT_EXTENSIONS",
    "TYPESCRIPT_EXTENSIONS",
    "JS_TS_EXTENSIONS",
    "CODE_EXTENSIONS",
    "DOCUMENTATION_EXTENSIONS",
    "CONFIG_EXTENSIONS",
    "filter_files_by_extension",
    "count_by_extension",
    "count_by_language",
    "get_test_files",
    "get_documentation_files",
    "calculate_ratio",
]

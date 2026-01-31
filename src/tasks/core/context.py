"""Task execution context with caching.

Provides TaskContext class for sharing cached file data across multiple task executions.
"""
from __future__ import annotations

import ast
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.tasks.shared import (
    collect_file_paths,
    _read_file_text,
)


@dataclass
class TaskContext:
    """
    Shared context for task execution to avoid redundant file scanning and parsing.
    
    This context caches:
    - File paths collected from storage or source
    - File contents already read
    - Parsed ASTs for Python files
    - Import graphs
    
    Usage:
        ctx = TaskContext(source_path="src")
        files = ctx.get_files()  # Cached after first call
        content = ctx.get_file_content(path)  # Cached per file
    """
    storage_path: Optional[str] = None
    source_path: Optional[str] = None
    
    # Cached data
    _files: Optional[List[str]] = field(default=None, repr=False)
    _file_contents: Dict[str, str] = field(default_factory=dict, repr=False)
    _parsed_asts: Dict[str, ast.AST] = field(default_factory=dict, repr=False)
    _import_graph: Optional[Dict[str, List[str]]] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def get_files(self) -> List[str]:
        """Get all file paths, with caching."""
        if self._files is None:
            with self._lock:
                if self._files is None:
                    self._files = collect_file_paths(self.storage_path, self.source_path)
        return self._files
    
    def get_file_content(self, path: str) -> str:
        """Get file content, with caching."""
        if path not in self._file_contents:
            self._file_contents[path] = _read_file_text(path)
        return self._file_contents[path]
    
    def get_parsed_ast(self, path: str) -> Optional[ast.AST]:
        """Get parsed Python AST, with caching."""
        if path not in self._parsed_asts:
            content = self.get_file_content(path)
            if content and path.endswith(".py"):
                try:
                    self._parsed_asts[path] = ast.parse(content)
                except Exception:
                    self._parsed_asts[path] = None  # type: ignore
        return self._parsed_asts.get(path)
    
    def get_import_graph(self) -> Dict[str, List[str]]:
        """Get import graph, with caching."""
        if self._import_graph is None:
            with self._lock:
                if self._import_graph is None:
                    from src.tasks.graph import _build_import_graph
                    self._import_graph = _build_import_graph(self.get_files())
        return self._import_graph
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._files = None
            self._file_contents.clear()
            self._parsed_asts.clear()
            self._import_graph = None


# Thread-local storage for the current context
_current_context: threading.local = threading.local()


def get_current_context() -> Optional[TaskContext]:
    """Get the current task context if one is active."""
    return getattr(_current_context, 'context', None)


def set_current_context(ctx: Optional[TaskContext]) -> None:
    """Set the current task context."""
    _current_context.context = ctx


class task_context:
    """
    Context manager for setting a shared TaskContext during task execution.
    
    Usage:
        with task_context(source_path="src") as ctx:
            result1 = task_a(source_path="src")  # Uses shared context
            result2 = task_b(source_path="src")  # Reuses cached files
    """
    def __init__(self, storage_path: Optional[str] = None, source_path: Optional[str] = None):
        self.ctx = TaskContext(storage_path=storage_path, source_path=source_path)
        self._previous: Optional[TaskContext] = None
    
    def __enter__(self) -> TaskContext:
        self._previous = get_current_context()
        set_current_context(self.ctx)
        return self.ctx
    
    def __exit__(self, *args) -> None:
        set_current_context(self._previous)

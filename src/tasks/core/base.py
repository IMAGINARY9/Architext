"""Base classes and utilities for task implementation.

This module provides common patterns and utilities to reduce boilerplate
in task implementations.
"""
from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from src.tasks.shared import (
    DEFAULT_EXTENSIONS,
    _progress,
    _read_file_text,
    collect_file_paths,
)
from src.tasks.core.context import get_current_context


# Extension groups for file filtering
PYTHON_EXTENSIONS: Set[str] = {".py"}
JAVASCRIPT_EXTENSIONS: Set[str] = {".js", ".jsx"}
TYPESCRIPT_EXTENSIONS: Set[str] = {".ts", ".tsx"}
JS_TS_EXTENSIONS: Set[str] = JAVASCRIPT_EXTENSIONS | TYPESCRIPT_EXTENSIONS
CODE_EXTENSIONS: Set[str] = PYTHON_EXTENSIONS | JS_TS_EXTENSIONS | {".java", ".go", ".rs", ".c", ".cpp", ".cs", ".rb", ".php"}
DOCUMENTATION_EXTENSIONS: Set[str] = {".md", ".rst", ".txt"}
CONFIG_EXTENSIONS: Set[str] = {".json", ".yml", ".yaml", ".toml", ".ini", ".cfg"}


@dataclass
class FileInfo:
    """Metadata about a file for analysis."""
    path: str
    extension: str
    language: str
    content: Optional[str] = None
    ast_tree: Optional[ast.AST] = None
    line_count: int = 0
    
    @classmethod
    def from_path(cls, path: str, load_content: bool = False) -> "FileInfo":
        """Create FileInfo from a file path."""
        p = Path(path)
        ext = p.suffix.lower()
        lang = DEFAULT_EXTENSIONS.get(ext, "Other")
        
        info = cls(
            path=path,
            extension=ext,
            language=lang,
        )
        
        if load_content:
            info.content = _read_file_text(path)
            if info.content:
                info.line_count = len(info.content.splitlines())
                if ext == ".py":
                    try:
                        info.ast_tree = ast.parse(info.content)
                    except Exception:
                        pass
        
        return info


@dataclass
class TaskResult:
    """Base result wrapper for tasks."""
    success: bool = True
    error: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard dictionary format."""
        if self.error:
            return {"success": False, "error": self.error}
        return {"success": True, **self.data}


class BaseTask(ABC):
    """
    Abstract base class for analysis tasks.
    
    Provides common functionality:
    - File collection and filtering
    - Progress reporting
    - Content and AST caching via TaskContext
    - Standardized error handling
    
    Subclasses should implement the `analyze()` method.
    
    Example:
        class MyTask(BaseTask):
            def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
                results = []
                for file in files:
                    if file.extension == ".py":
                        # Do analysis
                        pass
                return {"results": results}
        
        # Usage
        task = MyTask(source_path="src")
        result = task.run()
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        source_path: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        extensions: Optional[Set[str]] = None,
        load_content: bool = True,
    ):
        """
        Initialize the task.
        
        Args:
            storage_path: Path to ChromaDB storage
            source_path: Path to source directory  
            progress_callback: Optional callback for progress updates
            extensions: Set of file extensions to process (None = all)
            load_content: Whether to preload file content
        """
        self.storage_path = storage_path
        self.source_path = source_path
        self.progress_callback = progress_callback
        self.extensions = extensions
        self.load_content = load_content
        self._files: Optional[List[FileInfo]] = None
    
    def _report_progress(self, stage: str, message: str, **extra: Any) -> None:
        """Report progress to callback if provided."""
        _progress(self.progress_callback, {"stage": stage, "message": message, **extra})
    
    def _collect_files(self) -> List[str]:
        """Collect file paths, using context cache if available."""
        ctx = get_current_context()
        if ctx is not None:
            return ctx.get_files()
        return collect_file_paths(self.storage_path, self.source_path)
    
    def _get_file_content(self, path: str) -> str:
        """Get file content, using context cache if available."""
        ctx = get_current_context()
        if ctx is not None:
            return ctx.get_file_content(path)
        return _read_file_text(path)
    
    def get_files(self) -> List[FileInfo]:
        """
        Get list of files to analyze with metadata.
        
        Filters by extension if specified and caches the result.
        """
        if self._files is not None:
            return self._files
        
        self._report_progress("scan", "Collecting files")
        paths = self._collect_files()
        
        files = []
        for path in paths:
            ext = Path(path).suffix.lower()
            if self.extensions and ext not in self.extensions:
                continue
            
            info = FileInfo.from_path(path, load_content=self.load_content)
            files.append(info)
        
        self._files = files
        return files
    
    def get_python_files(self) -> List[FileInfo]:
        """Get only Python files with content and AST loaded."""
        return [f for f in self.get_files() if f.extension == ".py"]
    
    def get_js_ts_files(self) -> List[FileInfo]:
        """Get only JavaScript/TypeScript files."""
        return [f for f in self.get_files() if f.extension in JS_TS_EXTENSIONS]
    
    @abstractmethod
    def analyze(self, files: List[FileInfo]) -> Dict[str, Any]:
        """
        Perform the analysis.
        
        Subclasses must implement this method with the core analysis logic.
        
        Args:
            files: List of files to analyze (pre-filtered and loaded)
            
        Returns:
            Analysis results as a dictionary
        """
        pass
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the task.
        
        This is the main entry point that:
        1. Collects and filters files
        2. Calls analyze()
        3. Handles errors gracefully
        """
        try:
            files = self.get_files()
            self._report_progress("analyze", f"Analyzing {len(files)} files")
            return self.analyze(files)
        except Exception as e:
            return {"error": str(e), "success": False}


def filter_files_by_extension(
    paths: List[str],
    extensions: Set[str],
) -> List[str]:
    """Filter file paths by extension."""
    return [p for p in paths if Path(p).suffix.lower() in extensions]


def count_by_extension(paths: List[str]) -> Dict[str, int]:
    """Count files by extension."""
    from collections import Counter
    return dict(Counter(Path(p).suffix.lower() for p in paths))


def count_by_language(paths: List[str]) -> Dict[str, int]:
    """Count files by language."""
    from collections import Counter
    languages: Counter[str] = Counter()
    for path in paths:
        ext = Path(path).suffix.lower()
        lang = DEFAULT_EXTENSIONS.get(ext, "Other")
        languages[lang] += 1
    return dict(languages)


def get_test_files(paths: List[str]) -> List[str]:
    """Get files that appear to be tests."""
    test_patterns = {"test_", "_test", "test.", ".test.", ".spec."}
    return [
        p for p in paths
        if any(pattern in Path(p).name.lower() for pattern in test_patterns)
        or "/tests/" in p.replace("\\", "/").lower()
        or "/test/" in p.replace("\\", "/").lower()
    ]


def get_documentation_files(paths: List[str]) -> List[str]:
    """Get documentation files."""
    return [p for p in paths if Path(p).suffix.lower() in DOCUMENTATION_EXTENSIONS]


def calculate_ratio(count: int, total: int) -> float:
    """Calculate a ratio safely, avoiding division by zero."""
    if total == 0:
        return 0.0
    return count / total

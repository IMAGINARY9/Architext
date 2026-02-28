"""Task execution history and analytics.

This module tracks task executions over time, providing:
- Execution history with timing and status
- Analytics on task performance
- Trend analysis
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Literal
from collections import defaultdict


@dataclass
class TaskExecution:
    """Record of a single task execution."""
    task_name: str
    status: Literal["success", "error", "timeout"]
    started_at: float
    completed_at: float
    duration_seconds: float
    source_path: Optional[str] = None
    storage_path: Optional[str] = None
    error_message: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None
    cached: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_name": self.task_name,
            "status": self.status,
            "started_at": self.started_at,
            "started_at_iso": datetime.fromtimestamp(self.started_at).isoformat(),
            "completed_at": self.completed_at,
            "completed_at_iso": datetime.fromtimestamp(self.completed_at).isoformat(),
            "duration_seconds": round(self.duration_seconds, 3),
            "source_path": self.source_path,
            "storage_path": self.storage_path,
            "error_message": self.error_message,
            "result_summary": self.result_summary,
            "cached": self.cached,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskExecution":
        """Create from dictionary."""
        return cls(
            task_name=data["task_name"],
            status=data["status"],
            started_at=data["started_at"],
            completed_at=data["completed_at"],
            duration_seconds=data["duration_seconds"],
            source_path=data.get("source_path"),
            storage_path=data.get("storage_path"),
            error_message=data.get("error_message"),
            result_summary=data.get("result_summary"),
            cached=data.get("cached", False),
        )


@dataclass
class TaskAnalytics:
    """Analytics for a single task."""
    task_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    cached_executions: int = 0
    total_duration_seconds: float = 0.0
    min_duration_seconds: float = float("inf")
    max_duration_seconds: float = 0.0
    last_execution: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100
    
    @property
    def average_duration_seconds(self) -> float:
        """Calculate average execution duration."""
        non_cached = self.total_executions - self.cached_executions
        if non_cached == 0:
            return 0.0
        return self.total_duration_seconds / non_cached
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as a percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.cached_executions / self.total_executions) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_name": self.task_name,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "cached_executions": self.cached_executions,
            "success_rate_percent": round(self.success_rate, 2),
            "cache_hit_rate_percent": round(self.cache_hit_rate, 2),
            "average_duration_seconds": round(self.average_duration_seconds, 3),
            "min_duration_seconds": round(self.min_duration_seconds, 3) if self.min_duration_seconds != float("inf") else None,
            "max_duration_seconds": round(self.max_duration_seconds, 3),
            "last_execution_iso": datetime.fromtimestamp(self.last_execution).isoformat() if self.last_execution else None,
        }


class TaskExecutionHistory:
    """
    Tracks and persists task execution history.
    
    Features:
    - Records all task executions with timing
    - Persists history to disk (JSON)
    - Computes analytics per task and overall
    - Supports history trimming (max entries)
    
    Usage:
        history = TaskExecutionHistory()
        
        # Record an execution
        with history.track("analyze-structure", source_path="src") as tracker:
            result = run_task(...)
            tracker.set_result(result)
        
        # Get analytics
        stats = history.get_analytics()
        print(stats["analyze-structure"].average_duration_seconds)
    """
    
    def __init__(
        self,
        history_dir: Optional[str] = None,
        max_entries: int = 1000,
        enabled: bool = True,
    ):
        """
        Initialize history tracker.
        
        Args:
            history_dir: Directory to store history files. Defaults to ~/.architext/history
            max_entries: Maximum history entries to keep per task
            enabled: Whether history tracking is enabled
        """
        if history_dir is None:
            history_dir = str(Path.home() / ".architext" / "history")
        
        self.history_dir = Path(history_dir)
        self.max_entries = max_entries
        self.enabled = enabled
        self._lock = threading.Lock()
        self._history: Dict[str, List[TaskExecution]] = defaultdict(list)
        self._loaded = False
        
        if self.enabled:
            self.history_dir.mkdir(parents=True, exist_ok=True)
    
    def _history_file(self) -> Path:
        """Get the history file path."""
        return self.history_dir / "executions.json"
    
    def _load_history(self) -> None:
        """Load history from disk."""
        if self._loaded:
            return
        
        history_file = self._history_file()
        if not history_file.exists():
            self._loaded = True
            return
        
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for task_name, executions in data.items():
                self._history[task_name] = [
                    TaskExecution.from_dict(e) for e in executions
                ]
            self._loaded = True
        except Exception:
            self._loaded = True  # Mark as loaded even on error
    
    def _save_history(self) -> None:
        """Save history to disk."""
        if not self.enabled:
            return
        
        data = {
            task_name: [e.to_dict() for e in executions[-self.max_entries:]]
            for task_name, executions in self._history.items()
        }
        
        try:
            with open(self._history_file(), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Disk write failure shouldn't break the application
    
    def record(
        self,
        task_name: str,
        status: Literal["success", "error", "timeout"],
        started_at: float,
        completed_at: float,
        source_path: Optional[str] = None,
        storage_path: Optional[str] = None,
        error_message: Optional[str] = None,
        result_summary: Optional[Dict[str, Any]] = None,
        cached: bool = False,
    ) -> TaskExecution:
        """
        Record a task execution.
        
        Args:
            task_name: Name of the task
            status: Execution status
            started_at: Unix timestamp when execution started
            completed_at: Unix timestamp when execution completed
            source_path: Source path used
            storage_path: Storage path used
            error_message: Error message if failed
            result_summary: Summary of the result (not full result)
            cached: Whether result was from cache
            
        Returns:
            The recorded execution
        """
        if not self.enabled:
            return TaskExecution(
                task_name=task_name,
                status=status,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=completed_at - started_at,
                source_path=source_path,
                storage_path=storage_path,
                error_message=error_message,
                result_summary=result_summary,
                cached=cached,
            )
        
        execution = TaskExecution(
            task_name=task_name,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=completed_at - started_at,
            source_path=source_path,
            storage_path=storage_path,
            error_message=error_message,
            result_summary=result_summary,
            cached=cached,
        )
        
        with self._lock:
            self._load_history()
            self._history[task_name].append(execution)
            # Trim if over max entries
            if len(self._history[task_name]) > self.max_entries:
                self._history[task_name] = self._history[task_name][-self.max_entries:]
            self._save_history()
        
        return execution
    
    def track(
        self,
        task_name: str,
        source_path: Optional[str] = None,
        storage_path: Optional[str] = None,
    ) -> "ExecutionTracker":
        """
        Create a context manager to track task execution.
        
        Usage:
            with history.track("analyze-structure", source_path="src") as tracker:
                result = run_analysis()
                tracker.set_result(result)
        """
        return ExecutionTracker(
            history=self,
            task_name=task_name,
            source_path=source_path,
            storage_path=storage_path,
        )
    
    def get_history(
        self,
        task_name: Optional[str] = None,
        limit: int = 100,
        since: Optional[float] = None,
    ) -> List[TaskExecution]:
        """
        Get execution history.
        
        Args:
            task_name: Filter by task name (None for all)
            limit: Maximum entries to return
            since: Only return executions after this timestamp
            
        Returns:
            List of executions, newest first
        """
        with self._lock:
            self._load_history()
            
            if task_name:
                executions = list(self._history.get(task_name, []))
            else:
                executions = []
                for task_executions in self._history.values():
                    executions.extend(task_executions)
            
            # Filter by time
            if since:
                executions = [e for e in executions if e.started_at >= since]
            
            # Sort by start time (newest first)
            executions.sort(key=lambda e: e.started_at, reverse=True)
            
            return executions[:limit]
    
    def get_analytics(
        self,
        task_name: Optional[str] = None,
        since: Optional[float] = None,
    ) -> Dict[str, TaskAnalytics]:
        """
        Compute analytics for tasks.
        
        Args:
            task_name: Filter by task name (None for all)
            since: Only include executions after this timestamp
            
        Returns:
            Dictionary mapping task names to their analytics
        """
        with self._lock:
            self._load_history()
            
            analytics: Dict[str, TaskAnalytics] = {}
            
            task_names = [task_name] if task_name else list(self._history.keys())
            
            for name in task_names:
                executions = self._history.get(name, [])
                if since:
                    executions = [e for e in executions if e.started_at >= since]
                
                if not executions:
                    continue
                
                stats = TaskAnalytics(task_name=name)
                
                for e in executions:
                    stats.total_executions += 1
                    if e.status == "success":
                        stats.successful_executions += 1
                    else:
                        stats.failed_executions += 1
                    
                    if e.cached:
                        stats.cached_executions += 1
                    else:
                        # Only count non-cached for duration stats
                        stats.total_duration_seconds += e.duration_seconds
                        stats.min_duration_seconds = min(
                            stats.min_duration_seconds, e.duration_seconds
                        )
                        stats.max_duration_seconds = max(
                            stats.max_duration_seconds, e.duration_seconds
                        )
                    
                    if stats.last_execution is None or e.started_at > stats.last_execution:
                        stats.last_execution = e.started_at
                
                analytics[name] = stats
            
            return analytics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get an overall summary of task execution history."""
        analytics = self.get_analytics()
        
        total_executions = sum(a.total_executions for a in analytics.values())
        total_successful = sum(a.successful_executions for a in analytics.values())
        total_cached = sum(a.cached_executions for a in analytics.values())
        total_duration = sum(a.total_duration_seconds for a in analytics.values())
        
        # Find slowest and fastest tasks (by average)
        by_avg_duration = sorted(
            [(name, a.average_duration_seconds) for name, a in analytics.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        
        return {
            "total_tasks_tracked": len(analytics),
            "total_executions": total_executions,
            "total_successful": total_successful,
            "total_failed": total_executions - total_successful,
            "total_cached": total_cached,
            "overall_success_rate_percent": round(
                (total_successful / total_executions * 100) if total_executions > 0 else 0, 2
            ),
            "overall_cache_hit_rate_percent": round(
                (total_cached / total_executions * 100) if total_executions > 0 else 0, 2
            ),
            "total_compute_time_seconds": round(total_duration, 2),
            "slowest_tasks": by_avg_duration[:5],
            "fastest_tasks": by_avg_duration[-5:][::-1] if len(by_avg_duration) >= 5 else by_avg_duration[::-1],
            "per_task": {name: a.to_dict() for name, a in analytics.items()},
        }
    
    def clear(self, task_name: Optional[str] = None) -> int:
        """
        Clear history.
        
        Args:
            task_name: Clear only this task's history (None for all)
            
        Returns:
            Number of entries cleared
        """
        with self._lock:
            self._load_history()
            
            if task_name:
                count = len(self._history.get(task_name, []))
                self._history[task_name] = []
            else:
                count = sum(len(v) for v in self._history.values())
                self._history.clear()
            
            self._save_history()
            return count


class ExecutionTracker:
    """Context manager for tracking a single task execution."""
    
    def __init__(
        self,
        history: TaskExecutionHistory,
        task_name: str,
        source_path: Optional[str] = None,
        storage_path: Optional[str] = None,
    ):
        self.history = history
        self.task_name = task_name
        self.source_path = source_path
        self.storage_path = storage_path
        self.started_at: float = 0
        self.result_summary: Optional[Dict[str, Any]] = None
        self.cached: bool = False
        self.error_message: Optional[str] = None
    
    def __enter__(self) -> "ExecutionTracker":
        self.started_at = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        completed_at = time.time()
        
        if exc_type is not None:
            status = "error"
            self.error_message = str(exc_val)
        else:
            status = "success"
        
        self.history.record(
            task_name=self.task_name,
            status=status,
            started_at=self.started_at,
            completed_at=completed_at,
            source_path=self.source_path,
            storage_path=self.storage_path,
            error_message=self.error_message,
            result_summary=self.result_summary,
            cached=self.cached,
        )
    
    def set_result(self, result: Dict[str, Any], cached: bool = False) -> None:
        """Set the result summary and cache status."""
        # Extract a summary (avoid storing full results)
        self.result_summary = _extract_result_summary(result)
        self.cached = cached
    
    def set_error(self, error: str) -> None:
        """Set an error message."""
        self.error_message = error


def _extract_result_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a small summary from a task result."""
    summary = {}
    
    # Common summary fields
    for key in ["count", "total", "score", "findings", "issues", "files_analyzed"]:
        if key in result:
            value = result[key]
            if isinstance(value, (int, float, str, bool)):
                summary[key] = value
            elif isinstance(value, list):
                summary[f"{key}_count"] = len(value)
    
    # Include any explicitly small values
    for key, value in result.items():
        if isinstance(value, (int, float, bool)) and key not in summary:
            summary[key] = value
        elif key == "summary" and isinstance(value, dict):
            summary["summary"] = value
    
    return summary


# Global history instance
_global_history: Optional[TaskExecutionHistory] = None
_history_lock = threading.Lock()


def get_task_history(
    history_dir: Optional[str] = None,
    max_entries: int = 1000,
    enabled: bool = True,
) -> TaskExecutionHistory:
    """
    Get the global task history instance.
    
    Creates the history tracker on first call, reuses it on subsequent calls.
    """
    global _global_history
    
    with _history_lock:
        if _global_history is None:
            _global_history = TaskExecutionHistory(
                history_dir=history_dir,
                max_entries=max_entries,
                enabled=enabled,
            )
        return _global_history

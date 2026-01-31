"""Task execution metrics dashboard.

This module provides aggregated metrics and visualizations
for task execution history across the application.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.tasks.history import get_task_history, TaskExecution
from src.task_registry import TASK_REGISTRY, TASK_CATEGORIES


@dataclass
class ExecutionTrend:
    """Execution trend data for a time period."""
    period: str  # e.g., "2024-01-15", "2024-W03"
    total_executions: int
    successful: int
    failed: int
    cached: int
    total_duration_seconds: float
    
    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return (self.successful / self.total_executions) * 100
    
    @property
    def cache_hit_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return (self.cached / self.total_executions) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "total_executions": self.total_executions,
            "successful": self.successful,
            "failed": self.failed,
            "cached": self.cached,
            "success_rate": round(self.success_rate, 1),
            "cache_hit_rate": round(self.cache_hit_rate, 1),
            "total_duration_seconds": round(self.total_duration_seconds, 2),
        }


@dataclass
class TaskMetrics:
    """Metrics for a single task."""
    task_name: str
    total_executions: int
    successful: int
    failed: int
    cached: int
    total_duration_seconds: float
    average_duration_seconds: float
    last_execution: Optional[datetime]
    category: Optional[str]
    
    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return (self.successful / self.total_executions) * 100
    
    @property
    def cache_hit_rate(self) -> float:
        if self.total_executions == 0:
            return 0.0
        return (self.cached / self.total_executions) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "total_executions": self.total_executions,
            "successful": self.successful,
            "failed": self.failed,
            "cached": self.cached,
            "success_rate": round(self.success_rate, 1),
            "cache_hit_rate": round(self.cache_hit_rate, 1),
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "average_duration_seconds": round(self.average_duration_seconds, 3),
            "last_execution": self.last_execution.isoformat() if self.last_execution else None,
            "category": self.category,
        }


@dataclass
class DashboardMetrics:
    """Complete dashboard metrics."""
    # Summary
    total_executions: int
    total_tasks_run: int
    total_duration_seconds: float
    overall_success_rate: float
    overall_cache_hit_rate: float
    
    # Per-task metrics
    task_metrics: List[TaskMetrics]
    
    # Per-category metrics
    category_metrics: Dict[str, Dict[str, Any]]
    
    # Trends
    daily_trends: List[ExecutionTrend]
    
    # Top performers
    most_run_tasks: List[str]
    fastest_tasks: List[str]
    slowest_tasks: List[str]
    
    # Health indicators
    never_run_tasks: List[str]
    failing_tasks: List[str]  # Tasks with < 50% success rate
    
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_executions": self.total_executions,
                "total_tasks_run": self.total_tasks_run,
                "total_duration_seconds": round(self.total_duration_seconds, 2),
                "overall_success_rate": round(self.overall_success_rate, 1),
                "overall_cache_hit_rate": round(self.overall_cache_hit_rate, 1),
            },
            "task_metrics": [m.to_dict() for m in self.task_metrics],
            "category_metrics": self.category_metrics,
            "daily_trends": [t.to_dict() for t in self.daily_trends],
            "top_performers": {
                "most_run": self.most_run_tasks[:5],
                "fastest": self.fastest_tasks[:5],
                "slowest": self.slowest_tasks[:5],
            },
            "health": {
                "never_run_tasks": self.never_run_tasks,
                "failing_tasks": self.failing_tasks,
            },
            "generated_at": self.generated_at.isoformat(),
        }


class MetricsDashboard:
    """
    Dashboard for task execution metrics.
    
    Aggregates execution history into useful metrics and visualizations.
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        self._history = get_task_history()
        self._task_to_category = self._build_task_category_map()
    
    def _build_task_category_map(self) -> Dict[str, str]:
        """Build mapping from task name to category."""
        mapping = {}
        for category, tasks in TASK_CATEGORIES.items():
            for task in tasks:
                mapping[task] = category
        return mapping
    
    def get_dashboard(self, days: int = 30) -> DashboardMetrics:
        """
        Generate complete dashboard metrics.
        
        Args:
            days: Number of days to include in analysis
            
        Returns:
            DashboardMetrics with all metrics
        """
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_ts = cutoff.timestamp()
        all_history = self._history.get_history()
        
        # Filter to time range (started_at is a float timestamp)
        history = [h for h in all_history if h.started_at >= cutoff_ts]
        
        # Calculate summary metrics
        total = len(history)
        successful = sum(1 for h in history if h.status == "success")
        cached = sum(1 for h in history if h.cached)
        total_duration = sum(h.duration_seconds or 0 for h in history)
        
        tasks_run = set(h.task_name for h in history)
        
        # Per-task metrics
        task_metrics = self._calculate_task_metrics(history)
        
        # Category metrics
        category_metrics = self._calculate_category_metrics(history)
        
        # Daily trends
        daily_trends = self._calculate_daily_trends(history, days)
        
        # Top performers
        sorted_by_count = sorted(task_metrics, key=lambda m: m.total_executions, reverse=True)
        sorted_by_speed = sorted(
            [m for m in task_metrics if m.average_duration_seconds > 0],
            key=lambda m: m.average_duration_seconds
        )
        
        # Health indicators
        never_run = [t for t in TASK_REGISTRY if t not in tasks_run]
        failing = [m.task_name for m in task_metrics if m.success_rate < 50]
        
        return DashboardMetrics(
            total_executions=total,
            total_tasks_run=len(tasks_run),
            total_duration_seconds=total_duration,
            overall_success_rate=(successful / total * 100) if total > 0 else 0,
            overall_cache_hit_rate=(cached / total * 100) if total > 0 else 0,
            task_metrics=task_metrics,
            category_metrics=category_metrics,
            daily_trends=daily_trends,
            most_run_tasks=[m.task_name for m in sorted_by_count],
            fastest_tasks=[m.task_name for m in sorted_by_speed],
            slowest_tasks=[m.task_name for m in reversed(sorted_by_speed)],
            never_run_tasks=never_run,
            failing_tasks=failing,
        )
    
    def _calculate_task_metrics(self, history: List[TaskExecution]) -> List[TaskMetrics]:
        """Calculate per-task metrics."""
        task_data: Dict[str, List[TaskExecution]] = defaultdict(list)
        
        for h in history:
            task_data[h.task_name].append(h)
        
        metrics = []
        for task_name, executions in task_data.items():
            total = len(executions)
            successful = sum(1 for h in executions if h.status == "success")
            cached = sum(1 for h in executions if h.cached)
            durations = [h.duration_seconds for h in executions if h.duration_seconds]
            total_duration = sum(durations)
            avg_duration = total_duration / len(durations) if durations else 0
            
            # Convert float timestamp to datetime
            last_exec_ts = max((h.started_at for h in executions), default=None)
            last_exec = datetime.fromtimestamp(last_exec_ts) if last_exec_ts else None
            
            metrics.append(TaskMetrics(
                task_name=task_name,
                total_executions=total,
                successful=successful,
                failed=total - successful,
                cached=cached,
                total_duration_seconds=total_duration,
                average_duration_seconds=avg_duration,
                last_execution=last_exec,
                category=self._task_to_category.get(task_name),
            ))
        
        return sorted(metrics, key=lambda m: m.total_executions, reverse=True)
    
    def _calculate_category_metrics(
        self,
        history: List[TaskExecution],
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate per-category metrics."""
        category_data: Dict[str, List[TaskExecution]] = defaultdict(list)
        
        for h in history:
            category = self._task_to_category.get(h.task_name, "uncategorized")
            category_data[category].append(h)
        
        metrics = {}
        for category, executions in category_data.items():
            total = len(executions)
            successful = sum(1 for h in executions if h.status == "success")
            cached = sum(1 for h in executions if h.cached)
            total_duration = sum(h.duration_seconds or 0 for h in executions)
            
            tasks_run = set(h.task_name for h in executions)
            category_tasks = set(TASK_CATEGORIES.get(category, []))
            coverage = len(tasks_run) / len(category_tasks) * 100 if category_tasks else 0
            
            metrics[category] = {
                "total_executions": total,
                "successful": successful,
                "failed": total - successful,
                "cached": cached,
                "success_rate": round((successful / total * 100) if total > 0 else 0, 1),
                "cache_hit_rate": round((cached / total * 100) if total > 0 else 0, 1),
                "total_duration_seconds": round(total_duration, 2),
                "tasks_run": list(tasks_run),
                "task_coverage": round(coverage, 1),
            }
        
        return metrics
    
    def _calculate_daily_trends(
        self,
        history: List[TaskExecution],
        days: int,
    ) -> List[ExecutionTrend]:
        """Calculate daily execution trends."""
        daily_data: Dict[str, List[TaskExecution]] = defaultdict(list)
        
        for h in history:
            # started_at is a float timestamp
            day = datetime.fromtimestamp(h.started_at).strftime("%Y-%m-%d")
            daily_data[day].append(h)
        
        trends = []
        for day, executions in sorted(daily_data.items()):
            total = len(executions)
            successful = sum(1 for h in executions if h.status == "success")
            cached = sum(1 for h in executions if h.cached)
            total_duration = sum(h.duration_seconds or 0 for h in executions)
            
            trends.append(ExecutionTrend(
                period=day,
                total_executions=total,
                successful=successful,
                failed=total - successful,
                cached=cached,
                total_duration_seconds=total_duration,
            ))
        
        return trends
    
    def get_task_details(self, task_name: str, days: int = 30) -> Dict[str, Any]:
        """
        Get detailed metrics for a specific task.
        
        Args:
            task_name: Name of the task
            days: Number of days to include
            
        Returns:
            Detailed metrics dictionary
        """
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_ts = cutoff.timestamp()
        all_history = self._history.get_history(task_name=task_name)
        history = [h for h in all_history if h.started_at >= cutoff_ts]
        
        if not history:
            return {
                "task_name": task_name,
                "message": "No execution history found",
                "category": self._task_to_category.get(task_name),
            }
        
        total = len(history)
        successful = sum(1 for h in history if h.status == "success")
        cached = sum(1 for h in history if h.cached)
        durations = [h.duration_seconds for h in history if h.duration_seconds]
        
        # Calculate duration statistics
        avg_duration = sum(durations) / len(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        max_duration = max(durations) if durations else 0
        
        # Recent executions
        recent = sorted(history, key=lambda h: h.started_at, reverse=True)[:10]
        
        return {
            "task_name": task_name,
            "category": self._task_to_category.get(task_name),
            "metrics": {
                "total_executions": total,
                "successful": successful,
                "failed": total - successful,
                "cached": cached,
                "success_rate": round((successful / total * 100) if total > 0 else 0, 1),
                "cache_hit_rate": round((cached / total * 100) if total > 0 else 0, 1),
            },
            "duration": {
                "average_seconds": round(avg_duration, 3),
                "min_seconds": round(min_duration, 3),
                "max_seconds": round(max_duration, 3),
                "total_seconds": round(sum(durations), 2),
            },
            "recent_executions": [
                {
                    "started_at": datetime.fromtimestamp(h.started_at).isoformat(),
                    "status": h.status,
                    "duration_seconds": round(h.duration_seconds, 3) if h.duration_seconds else None,
                    "cached": h.cached,
                    "error": h.error_message,
                }
                for h in recent
            ],
        }


# Singleton instance
_dashboard: Optional[MetricsDashboard] = None


def get_metrics_dashboard() -> MetricsDashboard:
    """Get the singleton dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = MetricsDashboard()
    return _dashboard


def get_dashboard_metrics(days: int = 30) -> Dict[str, Any]:
    """
    Get dashboard metrics as dictionary.
    
    Convenience function for API usage.
    """
    dashboard = get_metrics_dashboard()
    return dashboard.get_dashboard(days).to_dict()


__all__ = [
    "ExecutionTrend",
    "TaskMetrics",
    "DashboardMetrics",
    "MetricsDashboard",
    "get_metrics_dashboard",
    "get_dashboard_metrics",
]

"""Compatibility shim — re-exports from src.tasks.orchestration.metrics.

All implementations live in src/tasks/orchestration/metrics.py.
Import from ``src.tasks.orchestration`` or ``src.tasks`` for the public API.
"""
from src.tasks.orchestration.metrics import *  # noqa: F401,F403
from src.tasks.orchestration.metrics import (  # explicit re-exports for type checkers
    ExecutionTrend,
    TaskMetrics,
    DashboardMetrics,
    MetricsDashboard,
    get_metrics_dashboard,
    get_dashboard_metrics,
)

__all__ = [
    "ExecutionTrend",
    "TaskMetrics",
    "DashboardMetrics",
    "MetricsDashboard",
    "get_metrics_dashboard",
    "get_dashboard_metrics",
]

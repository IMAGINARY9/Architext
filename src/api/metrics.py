"""Metrics dashboard endpoints.

Extracted from src/api/tasks.py — exposes execution metrics, trends,
per-task details, and category breakdowns.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException


def build_metrics_router() -> APIRouter:
    """Create an APIRouter with metrics dashboard endpoints."""
    router = APIRouter()

    @router.get("/tasks/metrics/dashboard")
    async def get_dashboard(days: int = 30) -> Dict[str, Any]:
        """Get the complete metrics dashboard.

        Args:
            days: Number of days to include in analysis (default: 30)
        """
        from src.tasks.orchestration.metrics import get_dashboard_metrics

        return get_dashboard_metrics(days)

    @router.get("/tasks/metrics/task/{task_name}")
    async def get_task_metrics(
        task_name: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get detailed metrics for a specific task.

        Args:
            task_name: Name of the task
            days: Number of days to include (default: 30)
        """
        from src.tasks.orchestration.metrics import get_metrics_dashboard
        from src.task_registry import TASK_REGISTRY

        if task_name not in TASK_REGISTRY:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found: {task_name}"
            )

        dashboard = get_metrics_dashboard()
        return dashboard.get_task_details(task_name, days)

    @router.get("/tasks/metrics/summary")
    async def get_metrics_summary(days: int = 30) -> Dict[str, Any]:
        """Get a summary of execution metrics.

        Returns only the summary portion of the dashboard for quick overview.
        """
        from src.tasks.orchestration.metrics import get_dashboard_metrics

        metrics = get_dashboard_metrics(days)
        return {
            "summary": metrics["summary"],
            "top_performers": metrics["top_performers"],
            "health": metrics["health"],
            "generated_at": metrics["generated_at"],
        }

    @router.get("/tasks/metrics/trends")
    async def get_execution_trends(days: int = 30) -> Dict[str, Any]:
        """Get daily execution trends.

        Returns execution counts and success rates by day.
        """
        from src.tasks.orchestration.metrics import get_dashboard_metrics

        metrics = get_dashboard_metrics(days)
        return {
            "daily_trends": metrics["daily_trends"],
            "days": days,
        }

    @router.get("/tasks/metrics/categories")
    async def get_category_metrics(days: int = 30) -> Dict[str, Any]:
        """Get metrics grouped by category.

        Shows execution stats for each task category.
        """
        from src.tasks.orchestration.metrics import get_dashboard_metrics

        metrics = get_dashboard_metrics(days)
        return {
            "category_metrics": metrics["category_metrics"],
            "days": days,
        }

    return router

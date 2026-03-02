"""Task scheduling endpoints.

Extracted from src/api/tasks.py — handles CRUD for task schedules,
enabling/disabling, manual execution, and scheduler lifecycle.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException


def build_scheduling_router() -> APIRouter:
    """Create an APIRouter with scheduling endpoints."""
    router = APIRouter()

    @router.get("/tasks/schedules")
    async def list_schedules() -> Dict[str, Any]:
        """List all scheduled tasks."""
        from src.tasks.orchestration.scheduler import get_task_scheduler

        scheduler = get_task_scheduler()
        schedules = scheduler.list_schedules()

        return {
            "schedules": [s.to_dict() for s in schedules],
            "count": len(schedules),
            "scheduler_running": scheduler.is_running,
        }

    @router.post("/tasks/schedules")
    async def create_schedule(
        request: Dict[str, Any] = Body(
            ...,
            examples=[
                {
                    "summary": "Run every 30 minutes",
                    "value": {
                        "task_name": "health-score",
                        "schedule_type": "interval",
                        "interval_minutes": 30,
                        "source_path": "./src",
                    },
                },
                {
                    "summary": "Daily at 2am",
                    "value": {
                        "task_name": "detect-anti-patterns",
                        "schedule_type": "cron",
                        "cron_minute": "0",
                        "cron_hour": "2",
                        "source_path": "./src",
                    },
                },
            ],
        ),
    ) -> Dict[str, Any]:
        """Create a scheduled task.

        Schedule types:
        - interval: Run every N minutes (specify interval_minutes)
        - cron: Cron-like schedule (specify cron_minute, cron_hour, cron_day_of_week)
        - once: Run once at specified time (specify run_at as ISO datetime)
        """
        from src.tasks.orchestration.scheduler import (
            ScheduleConfig, ScheduleType, get_task_scheduler
        )
        from src.task_registry import TASK_REGISTRY
        import uuid
        from datetime import datetime

        task_name = request.get("task_name")
        if not task_name:
            raise HTTPException(status_code=400, detail="task_name is required")

        if task_name not in TASK_REGISTRY:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found: {task_name}"
            )

        schedule_type_str = request.get("schedule_type", "interval")
        try:
            schedule_type = ScheduleType(schedule_type_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid schedule_type. Valid: {[t.value for t in ScheduleType]}"
            )

        config = ScheduleConfig(
            id=str(uuid.uuid4())[:8],
            task_name=task_name,
            schedule_type=schedule_type,
            enabled=request.get("enabled", True),
            interval_seconds=(
                request.get("interval_minutes", 60) * 60
                if schedule_type == ScheduleType.INTERVAL else None
            ),
            cron_minute=request.get("cron_minute"),
            cron_hour=request.get("cron_hour"),
            cron_day_of_week=request.get("cron_day_of_week"),
            run_at=(
                datetime.fromisoformat(request["run_at"])
                if request.get("run_at") else None
            ),
            source_path=request.get("source_path"),
            storage_path=request.get("storage_path"),
            task_params=request.get("params", {}),
        )

        scheduler = get_task_scheduler()
        created = scheduler.create_schedule(config)

        return {
            "status": "created",
            "schedule": created.to_dict(),
        }

    @router.get("/tasks/schedules/{schedule_id}")
    async def get_schedule(schedule_id: str) -> Dict[str, Any]:
        """Get details of a specific schedule."""
        from src.tasks.orchestration.scheduler import get_task_scheduler

        scheduler = get_task_scheduler()
        schedule = scheduler.get_schedule(schedule_id)

        if schedule is None:
            raise HTTPException(
                status_code=404,
                detail=f"Schedule not found: {schedule_id}"
            )

        return schedule.to_dict()

    @router.put("/tasks/schedules/{schedule_id}")
    async def update_schedule(
        schedule_id: str,
        request: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        """Update a schedule configuration."""
        from src.tasks.orchestration.scheduler import get_task_scheduler

        scheduler = get_task_scheduler()
        updated = scheduler.update_schedule(schedule_id, request)

        if updated is None:
            raise HTTPException(
                status_code=404,
                detail=f"Schedule not found: {schedule_id}"
            )

        return {
            "status": "updated",
            "schedule": updated.to_dict(),
        }

    @router.delete("/tasks/schedules/{schedule_id}")
    async def delete_schedule(schedule_id: str) -> Dict[str, Any]:
        """Delete a schedule."""
        from src.tasks.orchestration.scheduler import get_task_scheduler

        scheduler = get_task_scheduler()
        deleted = scheduler.delete_schedule(schedule_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Schedule not found: {schedule_id}"
            )

        return {"status": "deleted", "schedule_id": schedule_id}

    @router.post("/tasks/schedules/{schedule_id}/run")
    async def run_schedule_now(schedule_id: str) -> Dict[str, Any]:
        """Run a scheduled task immediately."""
        from src.tasks.orchestration.scheduler import get_task_scheduler

        scheduler = get_task_scheduler()
        execution = scheduler.run_now(schedule_id)

        if execution is None:
            raise HTTPException(
                status_code=404,
                detail=f"Schedule not found: {schedule_id}"
            )

        return {
            "status": "executed",
            "execution": execution.to_dict(),
        }

    @router.post("/tasks/schedules/{schedule_id}/enable")
    async def enable_schedule(schedule_id: str) -> Dict[str, Any]:
        """Enable a schedule."""
        from src.tasks.orchestration.scheduler import get_task_scheduler

        scheduler = get_task_scheduler()
        if not scheduler.enable_schedule(schedule_id):
            raise HTTPException(
                status_code=404,
                detail=f"Schedule not found: {schedule_id}"
            )

        return {"status": "enabled", "schedule_id": schedule_id}

    @router.post("/tasks/schedules/{schedule_id}/disable")
    async def disable_schedule(schedule_id: str) -> Dict[str, Any]:
        """Disable a schedule."""
        from src.tasks.orchestration.scheduler import get_task_scheduler

        scheduler = get_task_scheduler()
        if not scheduler.disable_schedule(schedule_id):
            raise HTTPException(
                status_code=404,
                detail=f"Schedule not found: {schedule_id}"
            )

        return {"status": "disabled", "schedule_id": schedule_id}

    @router.get("/tasks/schedules/{schedule_id}/executions")
    async def get_schedule_executions(
        schedule_id: str,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get execution history for a schedule."""
        from src.tasks.orchestration.scheduler import get_task_scheduler

        scheduler = get_task_scheduler()
        schedule = scheduler.get_schedule(schedule_id)

        if schedule is None:
            raise HTTPException(
                status_code=404,
                detail=f"Schedule not found: {schedule_id}"
            )

        executions = scheduler.get_executions(limit, schedule_id)

        return {
            "schedule_id": schedule_id,
            "executions": [e.to_dict() for e in executions],
            "count": len(executions),
        }

    @router.post("/tasks/scheduler/start")
    async def start_scheduler() -> Dict[str, Any]:
        """Start the background task scheduler."""
        from src.tasks.orchestration.scheduler import get_task_scheduler

        scheduler = get_task_scheduler()
        scheduler.start()

        return {
            "status": "started",
            "message": "Task scheduler is now running in background",
        }

    @router.post("/tasks/scheduler/stop")
    async def stop_scheduler() -> Dict[str, Any]:
        """Stop the background task scheduler."""
        from src.tasks.orchestration.scheduler import get_task_scheduler

        scheduler = get_task_scheduler()
        scheduler.stop()

        return {
            "status": "stopped",
            "message": "Task scheduler has been stopped",
        }

    return router

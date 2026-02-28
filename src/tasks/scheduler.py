"""Compatibility shim — re-exports from src.tasks.orchestration.scheduler.

All implementations live in src/tasks/orchestration/scheduler.py.
Import from ``src.tasks.orchestration`` or ``src.tasks`` for the public API.
"""
from src.tasks.orchestration.scheduler import *  # noqa: F401,F403
from src.tasks.orchestration.scheduler import (  # explicit re-exports for type checkers
    ScheduleType,
    ScheduleConfig,
    ScheduleExecution,
    TaskScheduler,
    get_task_scheduler,
    create_interval_schedule,
    create_cron_schedule,
    create_one_time_schedule,
)

__all__ = [
    "ScheduleType",
    "ScheduleConfig",
    "ScheduleExecution",
    "TaskScheduler",
    "get_task_scheduler",
    "create_interval_schedule",
    "create_cron_schedule",
    "create_one_time_schedule",
]

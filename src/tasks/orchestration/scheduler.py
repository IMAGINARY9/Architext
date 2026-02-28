"""Task scheduling and automation.

This module provides scheduling capabilities for automated task execution,
supporting cron-like schedules and interval-based triggers.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import re


class ScheduleType(str, Enum):
    """Types of schedule triggers."""
    INTERVAL = "interval"  # Run every N seconds/minutes/hours
    CRON = "cron"  # Cron-like schedule
    ONCE = "once"  # Run once at a specific time


@dataclass
class ScheduleConfig:
    """Configuration for a scheduled task."""
    id: str
    task_name: str
    schedule_type: ScheduleType
    enabled: bool = True
    
    # For interval schedules
    interval_seconds: Optional[int] = None
    
    # For cron schedules (simplified: minute, hour, day_of_week)
    cron_minute: Optional[str] = None  # 0-59 or *
    cron_hour: Optional[str] = None  # 0-23 or *
    cron_day_of_week: Optional[str] = None  # 0-6 (Mon-Sun) or *
    
    # For one-time schedules
    run_at: Optional[datetime] = None
    
    # Task parameters
    source_path: Optional[str] = None
    storage_path: Optional[str] = None
    task_params: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "task_name": self.task_name,
            "schedule_type": self.schedule_type.value,
            "enabled": self.enabled,
            "interval_seconds": self.interval_seconds,
            "cron_minute": self.cron_minute,
            "cron_hour": self.cron_hour,
            "cron_day_of_week": self.cron_day_of_week,
            "run_at": self.run_at.isoformat() if self.run_at else None,
            "source_path": self.source_path,
            "storage_path": self.storage_path,
            "task_params": self.task_params,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "last_error": self.last_error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduleConfig":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            task_name=data["task_name"],
            schedule_type=ScheduleType(data["schedule_type"]),
            enabled=data.get("enabled", True),
            interval_seconds=data.get("interval_seconds"),
            cron_minute=data.get("cron_minute"),
            cron_hour=data.get("cron_hour"),
            cron_day_of_week=data.get("cron_day_of_week"),
            run_at=datetime.fromisoformat(data["run_at"]) if data.get("run_at") else None,
            source_path=data.get("source_path"),
            storage_path=data.get("storage_path"),
            task_params=data.get("task_params", {}),
            last_run=datetime.fromisoformat(data["last_run"]) if data.get("last_run") else None,
            next_run=datetime.fromisoformat(data["next_run"]) if data.get("next_run") else None,
            run_count=data.get("run_count", 0),
            last_error=data.get("last_error"),
        )


@dataclass
class ScheduleExecution:
    """Record of a scheduled task execution."""
    schedule_id: str
    task_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"  # running, success, failed
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schedule_id": self.schedule_id,
            "task_name": self.task_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
            "result_summary": self.result_summary,
        }


class TaskScheduler:
    """
    Manages scheduled task execution.
    
    Supports interval-based and cron-like scheduling.
    """
    
    DEFAULT_STORAGE_PATH = Path.home() / ".architext" / "schedules"
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        task_runner: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        """
        Initialize the scheduler.
        
        Args:
            storage_path: Path to store schedule configurations
            task_runner: Function to run tasks (defaults to run_task from registry)
        """
        self.storage_path = storage_path or self.DEFAULT_STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._task_runner = task_runner
        self._schedules: Dict[str, ScheduleConfig] = {}
        self._executions: List[ScheduleExecution] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        self._load_schedules()
    
    def _get_task_runner(self) -> Callable[[str, Dict[str, Any]], Dict[str, Any]]:
        """Get the task runner function."""
        if self._task_runner:
            return self._task_runner
        
        # Lazy import to avoid circular dependency
        from src.task_registry import run_task
        return lambda name, params: run_task(name, **params)
    
    def _load_schedules(self) -> None:
        """Load schedules from storage."""
        config_file = self.storage_path / "schedules.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    data = json.load(f)
                for item in data.get("schedules", []):
                    schedule = ScheduleConfig.from_dict(item)
                    self._schedules[schedule.id] = schedule
            except Exception:
                pass  # Ignore load errors
    
    def _save_schedules(self) -> None:
        """Save schedules to storage."""
        config_file = self.storage_path / "schedules.json"
        data = {
            "schedules": [s.to_dict() for s in self._schedules.values()]
        }
        with open(config_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def create_schedule(self, config: ScheduleConfig) -> ScheduleConfig:
        """
        Create a new schedule.
        
        Args:
            config: Schedule configuration
            
        Returns:
            The created schedule
        """
        config.next_run = self._calculate_next_run(config)
        
        with self._lock:
            self._schedules[config.id] = config
            self._save_schedules()
        
        return config
    
    def delete_schedule(self, schedule_id: str) -> bool:
        """
        Delete a schedule.
        
        Args:
            schedule_id: ID of schedule to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if schedule_id in self._schedules:
                del self._schedules[schedule_id]
                self._save_schedules()
                return True
        return False
    
    def get_schedule(self, schedule_id: str) -> Optional[ScheduleConfig]:
        """Get a schedule by ID."""
        return self._schedules.get(schedule_id)
    
    def list_schedules(self) -> List[ScheduleConfig]:
        """List all schedules."""
        return list(self._schedules.values())
    
    def update_schedule(
        self,
        schedule_id: str,
        updates: Dict[str, Any],
    ) -> Optional[ScheduleConfig]:
        """
        Update a schedule configuration.
        
        Args:
            schedule_id: ID of schedule to update
            updates: Dictionary of fields to update
            
        Returns:
            Updated schedule or None if not found
        """
        with self._lock:
            if schedule_id not in self._schedules:
                return None
            
            schedule = self._schedules[schedule_id]
            
            if "enabled" in updates:
                schedule.enabled = updates["enabled"]
            if "interval_seconds" in updates:
                schedule.interval_seconds = updates["interval_seconds"]
            if "cron_minute" in updates:
                schedule.cron_minute = updates["cron_minute"]
            if "cron_hour" in updates:
                schedule.cron_hour = updates["cron_hour"]
            if "cron_day_of_week" in updates:
                schedule.cron_day_of_week = updates["cron_day_of_week"]
            if "source_path" in updates:
                schedule.source_path = updates["source_path"]
            if "storage_path" in updates:
                schedule.storage_path = updates["storage_path"]
            if "task_params" in updates:
                schedule.task_params = updates["task_params"]
            
            schedule.next_run = self._calculate_next_run(schedule)
            self._save_schedules()
            return schedule
    
    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a schedule."""
        return self.update_schedule(schedule_id, {"enabled": True}) is not None
    
    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a schedule."""
        return self.update_schedule(schedule_id, {"enabled": False}) is not None
    
    def _calculate_next_run(self, schedule: ScheduleConfig) -> Optional[datetime]:
        """Calculate the next run time for a schedule."""
        now = datetime.now()
        
        if schedule.schedule_type == ScheduleType.INTERVAL:
            if schedule.interval_seconds:
                if schedule.last_run:
                    return schedule.last_run + timedelta(seconds=schedule.interval_seconds)
                return now + timedelta(seconds=schedule.interval_seconds)
        
        elif schedule.schedule_type == ScheduleType.ONCE:
            if schedule.run_at and schedule.run_at > now:
                return schedule.run_at
            return None  # Already passed
        
        elif schedule.schedule_type == ScheduleType.CRON:
            return self._next_cron_time(schedule, now)
        
        return None
    
    def _next_cron_time(
        self,
        schedule: ScheduleConfig,
        after: datetime,
    ) -> Optional[datetime]:
        """Calculate next cron execution time (simplified)."""
        # Start from the next minute
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        # Search up to 7 days ahead
        for _ in range(7 * 24 * 60):
            if self._matches_cron(schedule, candidate):
                return candidate
            candidate += timedelta(minutes=1)
        
        return None
    
    def _matches_cron(self, schedule: ScheduleConfig, dt: datetime) -> bool:
        """Check if datetime matches cron pattern."""
        # Check minute
        if schedule.cron_minute and schedule.cron_minute != "*":
            if not self._matches_cron_field(schedule.cron_minute, dt.minute):
                return False
        
        # Check hour
        if schedule.cron_hour and schedule.cron_hour != "*":
            if not self._matches_cron_field(schedule.cron_hour, dt.hour):
                return False
        
        # Check day of week (0=Monday, 6=Sunday)
        if schedule.cron_day_of_week and schedule.cron_day_of_week != "*":
            if not self._matches_cron_field(schedule.cron_day_of_week, dt.weekday()):
                return False
        
        return True
    
    def _matches_cron_field(self, pattern: str, value: int) -> bool:
        """Check if value matches cron field pattern."""
        if pattern == "*":
            return True
        
        # Handle comma-separated values: "1,2,3"
        if "," in pattern:
            return value in [int(v) for v in pattern.split(",")]
        
        # Handle ranges: "1-5"
        if "-" in pattern:
            parts = pattern.split("-")
            if len(parts) == 2:
                return int(parts[0]) <= value <= int(parts[1])
        
        # Handle step: "*/5"
        if pattern.startswith("*/"):
            step = int(pattern[2:])
            return value % step == 0
        
        # Exact match
        return value == int(pattern)
    
    def run_now(self, schedule_id: str) -> Optional[ScheduleExecution]:
        """
        Run a scheduled task immediately.
        
        Args:
            schedule_id: ID of schedule to run
            
        Returns:
            Execution record or None if not found
        """
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return None
        
        return self._execute_schedule(schedule)
    
    def _execute_schedule(self, schedule: ScheduleConfig) -> ScheduleExecution:
        """Execute a scheduled task."""
        execution = ScheduleExecution(
            schedule_id=schedule.id,
            task_name=schedule.task_name,
            started_at=datetime.now(),
        )
        
        try:
            task_runner = self._get_task_runner()
            
            params = {
                **schedule.task_params,
            }
            if schedule.source_path:
                params["source_path"] = schedule.source_path
            if schedule.storage_path:
                params["storage_path"] = schedule.storage_path
            
            result = task_runner(schedule.task_name, params)
            
            execution.completed_at = datetime.now()
            execution.status = "success"
            execution.duration_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds()
            
            # Extract summary from result
            if isinstance(result, dict):
                if "error" in result:
                    execution.status = "failed"
                    execution.error = result["error"]
                else:
                    # Create a brief summary
                    execution.result_summary = {
                        k: v for k, v in result.items()
                        if k in ("count", "total", "score", "issues", "findings")
                    }
            
            schedule.last_error = None
            
        except Exception as e:
            execution.completed_at = datetime.now()
            execution.status = "failed"
            execution.error = str(e)
            execution.duration_seconds = (
                execution.completed_at - execution.started_at
            ).total_seconds()
            schedule.last_error = str(e)
        
        # Update schedule
        with self._lock:
            schedule.last_run = execution.started_at
            schedule.run_count += 1
            schedule.next_run = self._calculate_next_run(schedule)
            
            # For one-time schedules, disable after running
            if schedule.schedule_type == ScheduleType.ONCE:
                schedule.enabled = False
            
            self._save_schedules()
            
            self._executions.append(execution)
            if len(self._executions) > 500:
                self._executions = self._executions[-250:]
        
        return execution
    
    def start(self) -> None:
        """Start the scheduler background thread."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
    
    def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                self._check_schedules()
            except Exception:
                pass  # Don't let errors stop the loop
            
            # Wait 30 seconds between checks
            self._stop_event.wait(timeout=30)
    
    def _check_schedules(self) -> None:
        """Check and execute due schedules."""
        now = datetime.now()
        
        for schedule in self._schedules.values():
            if not schedule.enabled:
                continue
            
            if schedule.next_run and schedule.next_run <= now:
                # Execute in a separate thread to not block
                thread = threading.Thread(
                    target=self._execute_schedule,
                    args=(schedule,),
                    daemon=True,
                )
                thread.start()
    
    def get_executions(
        self,
        limit: int = 50,
        schedule_id: Optional[str] = None,
    ) -> List[ScheduleExecution]:
        """
        Get recent schedule executions.
        
        Args:
            limit: Maximum number to return
            schedule_id: Filter by schedule ID
            
        Returns:
            List of executions
        """
        with self._lock:
            executions = self._executions
            if schedule_id:
                executions = [e for e in executions if e.schedule_id == schedule_id]
            return executions[-limit:]
    
    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


# Helper functions for creating schedules

def create_interval_schedule(
    schedule_id: str,
    task_name: str,
    interval_minutes: int,
    source_path: Optional[str] = None,
    **task_params: Any,
) -> ScheduleConfig:
    """Create an interval-based schedule."""
    return ScheduleConfig(
        id=schedule_id,
        task_name=task_name,
        schedule_type=ScheduleType.INTERVAL,
        interval_seconds=interval_minutes * 60,
        source_path=source_path,
        task_params=task_params,
    )


def create_cron_schedule(
    schedule_id: str,
    task_name: str,
    minute: str = "*",
    hour: str = "*",
    day_of_week: str = "*",
    source_path: Optional[str] = None,
    **task_params: Any,
) -> ScheduleConfig:
    """Create a cron-style schedule."""
    return ScheduleConfig(
        id=schedule_id,
        task_name=task_name,
        schedule_type=ScheduleType.CRON,
        cron_minute=minute,
        cron_hour=hour,
        cron_day_of_week=day_of_week,
        source_path=source_path,
        task_params=task_params,
    )


def create_one_time_schedule(
    schedule_id: str,
    task_name: str,
    run_at: datetime,
    source_path: Optional[str] = None,
    **task_params: Any,
) -> ScheduleConfig:
    """Create a one-time schedule."""
    return ScheduleConfig(
        id=schedule_id,
        task_name=task_name,
        schedule_type=ScheduleType.ONCE,
        run_at=run_at,
        source_path=source_path,
        task_params=task_params,
    )


# Singleton instance
_scheduler: Optional[TaskScheduler] = None


def get_task_scheduler() -> TaskScheduler:
    """Get the singleton scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TaskScheduler()
    return _scheduler


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

"""Tests for task scheduling system."""
from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tasks.scheduler import (
    ScheduleConfig,
    ScheduleExecution,
    ScheduleType,
    TaskScheduler,
    create_cron_schedule,
    create_interval_schedule,
    create_one_time_schedule,
)


class TestScheduleConfig:
    """Tests for ScheduleConfig dataclass."""
    
    def test_create_interval_config(self):
        """Test creating interval schedule config."""
        config = ScheduleConfig(
            id="test-1",
            task_name="health-score",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=3600,
        )
        
        assert config.id == "test-1"
        assert config.task_name == "health-score"
        assert config.schedule_type == ScheduleType.INTERVAL
        assert config.interval_seconds == 3600
        assert config.enabled is True
    
    def test_create_cron_config(self):
        """Test creating cron schedule config."""
        config = ScheduleConfig(
            id="test-2",
            task_name="detect-anti-patterns",
            schedule_type=ScheduleType.CRON,
            cron_minute="0",
            cron_hour="2",
            cron_day_of_week="*",
        )
        
        assert config.schedule_type == ScheduleType.CRON
        assert config.cron_minute == "0"
        assert config.cron_hour == "2"
        assert config.cron_day_of_week == "*"
    
    def test_create_one_time_config(self):
        """Test creating one-time schedule config."""
        run_time = datetime.now() + timedelta(hours=1)
        config = ScheduleConfig(
            id="test-3",
            task_name="tech-stack",
            schedule_type=ScheduleType.ONCE,
            run_at=run_time,
        )
        
        assert config.schedule_type == ScheduleType.ONCE
        assert config.run_at == run_time
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ScheduleConfig(
            id="test-1",
            task_name="health-score",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=3600,
            source_path="./src",
        )
        
        data = config.to_dict()
        
        assert data["id"] == "test-1"
        assert data["task_name"] == "health-score"
        assert data["schedule_type"] == "interval"
        assert data["interval_seconds"] == 3600
        assert data["source_path"] == "./src"
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "id": "test-1",
            "task_name": "health-score",
            "schedule_type": "interval",
            "interval_seconds": 3600,
            "enabled": True,
        }
        
        config = ScheduleConfig.from_dict(data)
        
        assert config.id == "test-1"
        assert config.task_name == "health-score"
        assert config.schedule_type == ScheduleType.INTERVAL


class TestScheduleExecution:
    """Tests for ScheduleExecution dataclass."""
    
    def test_create_execution(self):
        """Test creating execution record."""
        execution = ScheduleExecution(
            schedule_id="test-1",
            task_name="health-score",
            started_at=datetime.now(),
        )
        
        assert execution.schedule_id == "test-1"
        assert execution.status == "running"
        assert execution.completed_at is None
    
    def test_to_dict(self):
        """Test converting execution to dictionary."""
        now = datetime.now()
        execution = ScheduleExecution(
            schedule_id="test-1",
            task_name="health-score",
            started_at=now,
            completed_at=now + timedelta(seconds=5),
            status="success",
            duration_seconds=5.0,
        )
        
        data = execution.to_dict()
        
        assert data["schedule_id"] == "test-1"
        assert data["status"] == "success"
        assert data["duration_seconds"] == 5.0


class TestTaskScheduler:
    """Tests for TaskScheduler class."""
    
    @pytest.fixture
    def scheduler(self, tmp_path):
        """Create a scheduler with temporary storage."""
        mock_runner = MagicMock(return_value={"score": 85})
        scheduler = TaskScheduler(
            storage_path=tmp_path / "schedules",
            task_runner=mock_runner,
        )
        yield scheduler
        scheduler.stop()
    
    def test_create_schedule(self, scheduler):
        """Test creating a schedule."""
        config = ScheduleConfig(
            id="test-1",
            task_name="health-score",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60,
        )
        
        created = scheduler.create_schedule(config)
        
        assert created.id == "test-1"
        assert created.next_run is not None
        assert scheduler.get_schedule("test-1") is not None
    
    def test_delete_schedule(self, scheduler):
        """Test deleting a schedule."""
        config = ScheduleConfig(
            id="test-1",
            task_name="health-score",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60,
        )
        scheduler.create_schedule(config)
        
        deleted = scheduler.delete_schedule("test-1")
        
        assert deleted is True
        assert scheduler.get_schedule("test-1") is None
    
    def test_delete_nonexistent_schedule(self, scheduler):
        """Test deleting non-existent schedule returns False."""
        deleted = scheduler.delete_schedule("nonexistent")
        assert deleted is False
    
    def test_list_schedules(self, scheduler):
        """Test listing schedules."""
        config1 = ScheduleConfig(
            id="test-1",
            task_name="health-score",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60,
        )
        config2 = ScheduleConfig(
            id="test-2",
            task_name="tech-stack",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=120,
        )
        
        scheduler.create_schedule(config1)
        scheduler.create_schedule(config2)
        
        schedules = scheduler.list_schedules()
        
        assert len(schedules) == 2
    
    def test_update_schedule(self, scheduler):
        """Test updating a schedule."""
        config = ScheduleConfig(
            id="test-1",
            task_name="health-score",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60,
        )
        scheduler.create_schedule(config)
        
        updated = scheduler.update_schedule("test-1", {"interval_seconds": 120})
        
        assert updated is not None
        assert updated.interval_seconds == 120
    
    def test_enable_disable_schedule(self, scheduler):
        """Test enabling and disabling schedules."""
        config = ScheduleConfig(
            id="test-1",
            task_name="health-score",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60,
            enabled=False,
        )
        scheduler.create_schedule(config)
        
        assert scheduler.enable_schedule("test-1") is True
        assert scheduler.get_schedule("test-1").enabled is True
        
        assert scheduler.disable_schedule("test-1") is True
        assert scheduler.get_schedule("test-1").enabled is False
    
    def test_run_now(self, scheduler):
        """Test running a schedule immediately."""
        config = ScheduleConfig(
            id="test-1",
            task_name="health-score",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60,
            source_path="./src",
        )
        scheduler.create_schedule(config)
        
        execution = scheduler.run_now("test-1")
        
        assert execution is not None
        assert execution.status == "success"
        assert execution.duration_seconds is not None
        
        # Check schedule was updated
        schedule = scheduler.get_schedule("test-1")
        assert schedule.run_count == 1
        assert schedule.last_run is not None
    
    def test_run_now_with_error(self, scheduler):
        """Test running a schedule that fails."""
        scheduler._task_runner = MagicMock(side_effect=Exception("Task failed"))
        
        config = ScheduleConfig(
            id="test-1",
            task_name="health-score",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60,
        )
        scheduler.create_schedule(config)
        
        execution = scheduler.run_now("test-1")
        
        assert execution.status == "failed"
        assert execution.error == "Task failed"
    
    def test_get_executions(self, scheduler):
        """Test getting execution history."""
        config = ScheduleConfig(
            id="test-1",
            task_name="health-score",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60,
        )
        scheduler.create_schedule(config)
        
        # Run multiple times
        scheduler.run_now("test-1")
        scheduler.run_now("test-1")
        
        executions = scheduler.get_executions(limit=10)
        
        assert len(executions) == 2
    
    def test_get_executions_filter_by_schedule(self, scheduler):
        """Test filtering executions by schedule ID."""
        config1 = ScheduleConfig(
            id="test-1",
            task_name="health-score",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60,
        )
        config2 = ScheduleConfig(
            id="test-2",
            task_name="tech-stack",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60,
        )
        scheduler.create_schedule(config1)
        scheduler.create_schedule(config2)
        
        scheduler.run_now("test-1")
        scheduler.run_now("test-2")
        scheduler.run_now("test-1")
        
        executions = scheduler.get_executions(schedule_id="test-1")
        
        assert len(executions) == 2
        assert all(e.schedule_id == "test-1" for e in executions)
    
    def test_persistence(self, tmp_path):
        """Test schedules are persisted to disk."""
        mock_runner = MagicMock(return_value={})
        storage = tmp_path / "schedules"
        
        # Create and save schedule
        scheduler1 = TaskScheduler(storage_path=storage, task_runner=mock_runner)
        config = ScheduleConfig(
            id="test-1",
            task_name="health-score",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60,
        )
        scheduler1.create_schedule(config)
        
        # Create new scheduler and check persistence
        scheduler2 = TaskScheduler(storage_path=storage, task_runner=mock_runner)
        
        assert scheduler2.get_schedule("test-1") is not None
        assert scheduler2.get_schedule("test-1").task_name == "health-score"
    
    def test_start_stop(self, scheduler):
        """Test starting and stopping the scheduler."""
        assert scheduler.is_running is False
        
        scheduler.start()
        assert scheduler.is_running is True
        
        scheduler.stop()
        time.sleep(0.1)  # Allow thread to stop
        assert scheduler.is_running is False


class TestCronMatching:
    """Tests for cron pattern matching."""
    
    @pytest.fixture
    def scheduler(self, tmp_path):
        """Create a scheduler for testing."""
        return TaskScheduler(
            storage_path=tmp_path / "schedules",
            task_runner=lambda n, p: {},
        )
    
    def test_matches_exact_minute(self, scheduler):
        """Test matching exact minute."""
        config = ScheduleConfig(
            id="test",
            task_name="test",
            schedule_type=ScheduleType.CRON,
            cron_minute="30",
            cron_hour="*",
        )
        
        dt = datetime(2024, 1, 1, 10, 30)
        assert scheduler._matches_cron(config, dt) is True
        
        dt = datetime(2024, 1, 1, 10, 31)
        assert scheduler._matches_cron(config, dt) is False
    
    def test_matches_wildcard(self, scheduler):
        """Test wildcard matching."""
        config = ScheduleConfig(
            id="test",
            task_name="test",
            schedule_type=ScheduleType.CRON,
            cron_minute="*",
            cron_hour="*",
        )
        
        dt = datetime(2024, 1, 1, 10, 30)
        assert scheduler._matches_cron(config, dt) is True
    
    def test_matches_range(self, scheduler):
        """Test range pattern matching."""
        config = ScheduleConfig(
            id="test",
            task_name="test",
            schedule_type=ScheduleType.CRON,
            cron_hour="9-17",  # 9am to 5pm
        )
        
        assert scheduler._matches_cron_field("9-17", 10) is True
        assert scheduler._matches_cron_field("9-17", 5) is False
    
    def test_matches_step(self, scheduler):
        """Test step pattern matching."""
        # Every 15 minutes
        assert scheduler._matches_cron_field("*/15", 0) is True
        assert scheduler._matches_cron_field("*/15", 15) is True
        assert scheduler._matches_cron_field("*/15", 30) is True
        assert scheduler._matches_cron_field("*/15", 10) is False
    
    def test_matches_list(self, scheduler):
        """Test comma-separated list matching."""
        assert scheduler._matches_cron_field("1,5,10", 5) is True
        assert scheduler._matches_cron_field("1,5,10", 3) is False


class TestHelperFunctions:
    """Tests for schedule creation helper functions."""
    
    def test_create_interval_schedule(self):
        """Test interval schedule helper."""
        config = create_interval_schedule(
            schedule_id="daily-scan",
            task_name="health-score",
            interval_minutes=30,
            source_path="./src",
        )
        
        assert config.id == "daily-scan"
        assert config.schedule_type == ScheduleType.INTERVAL
        assert config.interval_seconds == 1800  # 30 * 60
        assert config.source_path == "./src"
    
    def test_create_cron_schedule(self):
        """Test cron schedule helper."""
        config = create_cron_schedule(
            schedule_id="nightly-scan",
            task_name="detect-anti-patterns",
            minute="0",
            hour="2",
            day_of_week="1-5",  # Weekdays
            source_path="./src",
        )
        
        assert config.id == "nightly-scan"
        assert config.schedule_type == ScheduleType.CRON
        assert config.cron_minute == "0"
        assert config.cron_hour == "2"
        assert config.cron_day_of_week == "1-5"
    
    def test_create_one_time_schedule(self):
        """Test one-time schedule helper."""
        run_time = datetime(2024, 12, 31, 23, 59)
        config = create_one_time_schedule(
            schedule_id="eoy-scan",
            task_name="synthesis-roadmap",
            run_at=run_time,
            source_path="./project",
        )
        
        assert config.id == "eoy-scan"
        assert config.schedule_type == ScheduleType.ONCE
        assert config.run_at == run_time


class TestNextRunCalculation:
    """Tests for next run time calculation."""
    
    @pytest.fixture
    def scheduler(self, tmp_path):
        """Create a scheduler for testing."""
        return TaskScheduler(
            storage_path=tmp_path / "schedules",
            task_runner=lambda n, p: {},
        )
    
    def test_interval_next_run_no_previous(self, scheduler):
        """Test next run for interval with no previous run."""
        config = ScheduleConfig(
            id="test",
            task_name="test",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60,
        )
        
        next_run = scheduler._calculate_next_run(config)
        
        assert next_run is not None
        assert next_run > datetime.now()
    
    def test_interval_next_run_with_previous(self, scheduler):
        """Test next run for interval with previous run."""
        last_run = datetime.now() - timedelta(seconds=30)
        config = ScheduleConfig(
            id="test",
            task_name="test",
            schedule_type=ScheduleType.INTERVAL,
            interval_seconds=60,
            last_run=last_run,
        )
        
        next_run = scheduler._calculate_next_run(config)
        
        assert next_run is not None
        assert next_run == last_run + timedelta(seconds=60)
    
    def test_once_next_run_future(self, scheduler):
        """Test next run for one-time schedule in future."""
        run_at = datetime.now() + timedelta(hours=1)
        config = ScheduleConfig(
            id="test",
            task_name="test",
            schedule_type=ScheduleType.ONCE,
            run_at=run_at,
        )
        
        next_run = scheduler._calculate_next_run(config)
        
        assert next_run == run_at
    
    def test_once_next_run_past(self, scheduler):
        """Test next run for one-time schedule in past returns None."""
        run_at = datetime.now() - timedelta(hours=1)
        config = ScheduleConfig(
            id="test",
            task_name="test",
            schedule_type=ScheduleType.ONCE,
            run_at=run_at,
        )
        
        next_run = scheduler._calculate_next_run(config)
        
        assert next_run is None
    
    def test_one_time_disabled_after_run(self, scheduler):
        """Test one-time schedule is disabled after running."""
        run_at = datetime.now() + timedelta(minutes=1)
        config = ScheduleConfig(
            id="test-once",
            task_name="health-score",
            schedule_type=ScheduleType.ONCE,
            run_at=run_at,
        )
        
        scheduler.create_schedule(config)
        scheduler.run_now("test-once")
        
        schedule = scheduler.get_schedule("test-once")
        assert schedule.enabled is False

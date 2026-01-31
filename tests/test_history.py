"""Tests for task execution history and analytics."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from src.tasks.history import (
    TaskExecution,
    TaskAnalytics,
    TaskExecutionHistory,
    ExecutionTracker,
    get_task_history,
)


class TestTaskExecution:
    """Tests for TaskExecution dataclass."""
    
    def test_create_execution(self):
        """Test creating a task execution record."""
        now = time.time()
        execution = TaskExecution(
            task_name="test-task",
            status="success",
            started_at=now,
            completed_at=now + 1.5,
            duration_seconds=1.5,
        )
        assert execution.task_name == "test-task"
        assert execution.duration_seconds == 1.5
        assert execution.status == "success"
        assert execution.cached is False
        assert execution.error_message is None
    
    def test_execution_with_error(self):
        """Test execution record with error."""
        now = time.time()
        execution = TaskExecution(
            task_name="test-task",
            status="error",
            started_at=now,
            completed_at=now + 0.5,
            duration_seconds=0.5,
            error_message="Something went wrong",
        )
        assert execution.status == "error"
        assert execution.error_message == "Something went wrong"
    
    def test_execution_with_cache_hit(self):
        """Test execution record with cache hit."""
        now = time.time()
        execution = TaskExecution(
            task_name="test-task",
            status="success",
            started_at=now,
            completed_at=now + 0.01,
            duration_seconds=0.01,
            cached=True,
        )
        assert execution.cached is True
    
    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        now = time.time()
        execution = TaskExecution(
            task_name="test-task",
            status="success",
            started_at=now,
            completed_at=now + 2.5,
            duration_seconds=2.5,
            cached=True,
            source_path="/test",
        )
        
        data = execution.to_dict()
        restored = TaskExecution.from_dict(data)
        
        assert restored.task_name == execution.task_name
        assert restored.status == execution.status
        assert restored.duration_seconds == execution.duration_seconds
        assert restored.cached == execution.cached
        assert restored.source_path == execution.source_path


class TestTaskAnalytics:
    """Tests for TaskAnalytics dataclass."""
    
    def test_empty_analytics(self):
        """Test analytics with no executions."""
        analytics = TaskAnalytics(task_name="test-task")
        
        assert analytics.total_executions == 0
        assert analytics.successful_executions == 0
        assert analytics.failed_executions == 0
        assert analytics.cached_executions == 0
        assert analytics.success_rate == 0.0
        assert analytics.cache_hit_rate == 0.0
        assert analytics.average_duration_seconds == 0.0
    
    def test_analytics_calculations(self):
        """Test analytics with execution data."""
        analytics = TaskAnalytics(
            task_name="test-task",
            total_executions=10,
            successful_executions=8,
            failed_executions=2,
            cached_executions=3,
            total_duration_seconds=35.0,  # For 7 non-cached
            min_duration_seconds=1.0,
            max_duration_seconds=10.0,
        )
        
        assert analytics.success_rate == 80.0
        assert analytics.cache_hit_rate == 30.0
        assert analytics.average_duration_seconds == 5.0  # 35 / 7


class TestTaskExecutionHistory:
    """Tests for TaskExecutionHistory class."""
    
    @pytest.fixture
    def temp_history(self, tmp_path):
        """Create a history instance with temp storage."""
        return TaskExecutionHistory(history_dir=str(tmp_path))
    
    def test_record_execution(self, temp_history):
        """Test recording a single execution."""
        now = time.time()
        temp_history.record(
            task_name="test-task",
            status="success",
            started_at=now,
            completed_at=now + 1.5,
        )
        
        history = temp_history.get_history("test-task")
        assert len(history) == 1
        assert history[0].task_name == "test-task"
        assert history[0].status == "success"
    
    def test_record_multiple_executions(self, temp_history):
        """Test recording multiple executions."""
        now = time.time()
        for i in range(5):
            temp_history.record(
                task_name="test-task",
                status="success" if i % 2 == 0 else "error",
                started_at=now + i,
                completed_at=now + i + 1.0,
            )
        
        history = temp_history.get_history("test-task")
        assert len(history) == 5
    
    def test_get_history_with_limit(self, temp_history):
        """Test getting history with limit."""
        now = time.time()
        for i in range(10):
            temp_history.record(
                task_name="test-task",
                status="success",
                started_at=now + i,
                completed_at=now + i + 1.0,
            )
        
        history = temp_history.get_history("test-task", limit=5)
        assert len(history) == 5
    
    def test_get_history_all_tasks(self, temp_history):
        """Test getting history across all tasks."""
        now = time.time()
        temp_history.record("task-a", "success", now, now + 1)
        temp_history.record("task-b", "success", now + 1, now + 2)
        temp_history.record("task-c", "success", now + 2, now + 3)
        
        history = temp_history.get_history()
        assert len(history) == 3
    
    def test_get_analytics(self, temp_history):
        """Test computing analytics for a task."""
        now = time.time()
        temp_history.record("test-task", "success", now, now + 1)
        temp_history.record("test-task", "success", now + 1, now + 3)
        temp_history.record("test-task", "error", now + 3, now + 6)
        temp_history.record("test-task", "success", now + 6, now + 6.5, cached=True)
        
        analytics = temp_history.get_analytics("test-task")
        
        assert "test-task" in analytics
        stats = analytics["test-task"]
        assert stats.total_executions == 4
        assert stats.successful_executions == 3
        assert stats.failed_executions == 1
        assert stats.cached_executions == 1
    
    def test_get_analytics_all_tasks(self, temp_history):
        """Test getting analytics for all tasks."""
        now = time.time()
        temp_history.record("task-a", "success", now, now + 1)
        temp_history.record("task-b", "success", now + 1, now + 2)
        temp_history.record("task-c", "error", now + 2, now + 3)
        
        all_analytics = temp_history.get_analytics()
        
        assert len(all_analytics) == 3
        assert "task-a" in all_analytics
        assert "task-b" in all_analytics
        assert "task-c" in all_analytics
    
    def test_clear_history_all(self, temp_history):
        """Test clearing all history."""
        now = time.time()
        temp_history.record("task-a", "success", now, now + 1)
        temp_history.record("task-b", "success", now + 1, now + 2)
        
        count = temp_history.clear()
        
        assert count == 2
        assert len(temp_history.get_history()) == 0
    
    def test_clear_history_specific_task(self, temp_history):
        """Test clearing history for specific task."""
        now = time.time()
        temp_history.record("task-a", "success", now, now + 1)
        temp_history.record("task-a", "success", now + 1, now + 2)
        temp_history.record("task-b", "success", now + 2, now + 3)
        
        count = temp_history.clear(task_name="task-a")
        
        assert count == 2
        assert len(temp_history.get_history("task-a")) == 0
        assert len(temp_history.get_history("task-b")) == 1
    
    def test_persistence(self, tmp_path):
        """Test that history persists to disk."""
        now = time.time()
        # Create history and add records
        history1 = TaskExecutionHistory(history_dir=str(tmp_path))
        history1.record("test-task", "success", now, now + 1)
        history1.record("test-task", "error", now + 1, now + 2)
        
        # Create new instance from same path
        history2 = TaskExecutionHistory(history_dir=str(tmp_path))
        
        loaded = history2.get_history("test-task")
        assert len(loaded) == 2
    
    def test_max_entries(self, tmp_path):
        """Test that history respects max entries limit."""
        history = TaskExecutionHistory(
            history_dir=str(tmp_path),
            max_entries=5,
        )
        
        now = time.time()
        for i in range(10):
            history.record("test-task", "success", now + i, now + i + 1)
        
        stored = history.get_history("test-task")
        assert len(stored) == 5


class TestExecutionTracker:
    """Tests for ExecutionTracker context manager."""
    
    @pytest.fixture
    def temp_history(self, tmp_path):
        """Create a history instance with temp storage."""
        return TaskExecutionHistory(history_dir=str(tmp_path))
    
    def test_successful_execution(self, temp_history):
        """Test tracking a successful execution."""
        with temp_history.track("test-task") as tracker:
            time.sleep(0.01)  # Simulate some work
            tracker.set_result({"data": "value"})
        
        history = temp_history.get_history("test-task")
        assert len(history) == 1
        assert history[0].status == "success"
        assert history[0].duration_seconds > 0
    
    def test_failed_execution(self, temp_history):
        """Test tracking a failed execution via set_error before exception."""
        # The ExecutionTracker captures the exception message in __exit__
        # To test set_error specifically, we need a different approach
        try:
            with temp_history.track("test-task") as tracker:
                tracker.set_error("Test error")
                raise ValueError("Test error")  # Use same message
        except ValueError:
            pass  # Expected
        
        history = temp_history.get_history("test-task")
        assert len(history) == 1
        assert history[0].status == "error"
        assert "Test error" in history[0].error_message
    
    def test_exception_handling(self, temp_history):
        """Test that exceptions are tracked."""
        with pytest.raises(ValueError):
            with temp_history.track("test-task"):
                raise ValueError("Test exception")
        
        history = temp_history.get_history("test-task")
        assert len(history) == 1
        assert history[0].status == "error"
        assert "Test exception" in history[0].error_message
    
    def test_with_source_path(self, temp_history):
        """Test tracking with source path."""
        with temp_history.track(
            "test-task",
            source_path="/test",
        ) as tracker:
            tracker.set_result({})
        
        history = temp_history.get_history("test-task")
        assert history[0].source_path == "/test"


class TestSingleton:
    """Tests for singleton pattern."""
    
    def test_get_task_history_singleton(self, tmp_path):
        """Test that get_task_history returns singleton."""
        # Reset singleton for test
        import src.tasks.history as history_module
        history_module._task_history_instance = None
        
        # Get singleton
        h1 = get_task_history()
        h2 = get_task_history()
        
        # Same instance
        assert h1 is h2

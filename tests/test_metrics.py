"""Tests for task metrics dashboard."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.tasks.metrics import (
    ExecutionTrend,
    TaskMetrics,
    DashboardMetrics,
    MetricsDashboard,
    get_metrics_dashboard,
    get_dashboard_metrics,
)
from src.tasks.history import TaskExecution


class TestExecutionTrend:
    """Tests for ExecutionTrend dataclass."""
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        trend = ExecutionTrend(
            period="2024-01-15",
            total_executions=10,
            successful=8,
            failed=2,
            cached=3,
            total_duration_seconds=25.5,
        )
        
        assert trend.success_rate == 80.0
    
    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        trend = ExecutionTrend(
            period="2024-01-15",
            total_executions=10,
            successful=8,
            failed=2,
            cached=4,
            total_duration_seconds=20.0,
        )
        
        assert trend.cache_hit_rate == 40.0
    
    def test_zero_executions_rates(self):
        """Test rates are 0 when no executions."""
        trend = ExecutionTrend(
            period="2024-01-15",
            total_executions=0,
            successful=0,
            failed=0,
            cached=0,
            total_duration_seconds=0,
        )
        
        assert trend.success_rate == 0.0
        assert trend.cache_hit_rate == 0.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        trend = ExecutionTrend(
            period="2024-01-15",
            total_executions=10,
            successful=9,
            failed=1,
            cached=5,
            total_duration_seconds=30.123,
        )
        
        d = trend.to_dict()
        
        assert d["period"] == "2024-01-15"
        assert d["total_executions"] == 10
        assert d["success_rate"] == 90.0
        assert d["cache_hit_rate"] == 50.0
        assert d["total_duration_seconds"] == 30.12


class TestTaskMetrics:
    """Tests for TaskMetrics dataclass."""
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        metrics = TaskMetrics(
            task_name="analyze-structure",
            total_executions=20,
            successful=18,
            failed=2,
            cached=10,
            total_duration_seconds=100.0,
            average_duration_seconds=5.0,
            last_execution=datetime.now(),
            category="analysis",
        )
        
        assert metrics.success_rate == 90.0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime(2024, 1, 15, 12, 0)
        metrics = TaskMetrics(
            task_name="test-task",
            total_executions=10,
            successful=9,
            failed=1,
            cached=5,
            total_duration_seconds=50.5,
            average_duration_seconds=5.05,
            last_execution=now,
            category="quality",
        )
        
        d = metrics.to_dict()
        
        assert d["task_name"] == "test-task"
        assert d["success_rate"] == 90.0
        assert d["cache_hit_rate"] == 50.0
        assert d["last_execution"] == "2024-01-15T12:00:00"
        assert d["category"] == "quality"


class TestDashboardMetrics:
    """Tests for DashboardMetrics dataclass."""
    
    def test_to_dict_structure(self):
        """Test that to_dict has expected structure."""
        metrics = DashboardMetrics(
            total_executions=100,
            total_tasks_run=10,
            total_duration_seconds=500.0,
            overall_success_rate=95.0,
            overall_cache_hit_rate=40.0,
            task_metrics=[],
            category_metrics={},
            daily_trends=[],
            most_run_tasks=["a", "b"],
            fastest_tasks=["c"],
            slowest_tasks=["d"],
            never_run_tasks=["e"],
            failing_tasks=[],
        )
        
        d = metrics.to_dict()
        
        assert "summary" in d
        assert d["summary"]["total_executions"] == 100
        assert d["summary"]["overall_success_rate"] == 95.0
        assert "task_metrics" in d
        assert "category_metrics" in d
        assert "daily_trends" in d
        assert "top_performers" in d
        assert "health" in d
        assert "generated_at" in d


class TestMetricsDashboard:
    """Tests for MetricsDashboard class."""
    
    @patch("src.tasks.metrics.get_task_history")
    def test_get_dashboard_returns_metrics(self, mock_history):
        """Test that get_dashboard returns DashboardMetrics."""
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = []
        mock_history.return_value = mock_history_instance
        
        dashboard = MetricsDashboard()
        metrics = dashboard.get_dashboard(days=7)
        
        assert isinstance(metrics, DashboardMetrics)
    
    @patch("src.tasks.metrics.get_task_history")
    def test_get_dashboard_with_history(self, mock_history):
        """Test dashboard with execution history."""
        now = datetime.now().timestamp()
        executions = [
            TaskExecution(
                task_name="analyze-structure",
                status="success",
                started_at=now - 3600,
                completed_at=now - 3540,
                duration_seconds=60.0,
                cached=False,
            ),
            TaskExecution(
                task_name="tech-stack",
                status="success",
                started_at=now - 7200,
                completed_at=now - 7080,
                duration_seconds=120.0,
                cached=True,
            ),
            TaskExecution(
                task_name="analyze-structure",
                status="error",
                started_at=now - 10800,
                completed_at=now - 10770,
                duration_seconds=30.0,
                cached=False,
                error_message="Test error",
            ),
        ]
        
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = executions
        mock_history.return_value = mock_history_instance
        
        dashboard = MetricsDashboard()
        metrics = dashboard.get_dashboard(days=7)
        
        assert metrics.total_executions == 3
        assert metrics.total_tasks_run == 2  # 2 unique tasks
        assert len(metrics.task_metrics) == 2
    
    @patch("src.tasks.metrics.get_task_history")
    def test_get_task_details(self, mock_history):
        """Test getting details for specific task."""
        now = datetime.now().timestamp()
        executions = [
            TaskExecution(
                task_name="analyze-structure",
                status="success",
                started_at=now - 3600,
                completed_at=now,
                duration_seconds=60.0,
                cached=False,
            ),
        ]
        
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = executions
        mock_history.return_value = mock_history_instance
        
        dashboard = MetricsDashboard()
        details = dashboard.get_task_details("analyze-structure", days=7)
        
        assert details["task_name"] == "analyze-structure"
        assert "metrics" in details
        assert "duration" in details
        assert "recent_executions" in details
    
    @patch("src.tasks.metrics.get_task_history")
    def test_get_task_details_no_history(self, mock_history):
        """Test getting details when no history exists."""
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = []
        mock_history.return_value = mock_history_instance
        
        dashboard = MetricsDashboard()
        details = dashboard.get_task_details("nonexistent-task", days=7)
        
        assert details["task_name"] == "nonexistent-task"
        assert "message" in details
    
    @patch("src.tasks.metrics.get_task_history")
    def test_calculates_category_metrics(self, mock_history):
        """Test that category metrics are calculated."""
        now = datetime.now().timestamp()
        executions = [
            TaskExecution(
                task_name="health-score",  # quality category
                status="success",
                started_at=now - 3600,
                completed_at=now,
                duration_seconds=10.0,
                cached=False,
            ),
        ]
        
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = executions
        mock_history.return_value = mock_history_instance
        
        dashboard = MetricsDashboard()
        metrics = dashboard.get_dashboard(days=7)
        
        assert len(metrics.category_metrics) > 0
    
    @patch("src.tasks.metrics.get_task_history")
    def test_calculates_daily_trends(self, mock_history):
        """Test that daily trends are calculated."""
        now = datetime.now().timestamp()
        yesterday = now - 86400  # 24 hours ago
        
        executions = [
            TaskExecution(
                task_name="task1",
                status="success",
                started_at=now,
                completed_at=now,
                duration_seconds=5.0,
                cached=False,
            ),
            TaskExecution(
                task_name="task2",
                status="success",
                started_at=yesterday,
                completed_at=yesterday,
                duration_seconds=5.0,
                cached=False,
            ),
        ]
        
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = executions
        mock_history.return_value = mock_history_instance
        
        dashboard = MetricsDashboard()
        metrics = dashboard.get_dashboard(days=7)
        
        assert len(metrics.daily_trends) == 2  # Two different days
    
    @patch("src.tasks.metrics.get_task_history")
    def test_identifies_failing_tasks(self, mock_history):
        """Test that failing tasks are identified."""
        now = datetime.now().timestamp()
        
        # Create task with low success rate
        executions = [
            TaskExecution(
                task_name="failing-task",
                status="error",
                started_at=now - (3600 * i),
                completed_at=now - (3600 * i),
                duration_seconds=5.0,
                cached=False,
            )
            for i in range(3)  # 3 failures
        ]
        
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = executions
        mock_history.return_value = mock_history_instance
        
        dashboard = MetricsDashboard()
        metrics = dashboard.get_dashboard(days=7)
        
        assert "failing-task" in metrics.failing_tasks


class TestSingletonAndConvenienceFunctions:
    """Tests for singleton and convenience functions."""
    
    @patch("src.tasks.metrics.get_task_history")
    def test_get_metrics_dashboard_singleton(self, mock_history):
        """Test that singleton returns same instance."""
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = []
        mock_history.return_value = mock_history_instance
        
        # Reset singleton
        import src.tasks.metrics as metrics_module
        metrics_module._dashboard = None
        
        dashboard1 = get_metrics_dashboard()
        dashboard2 = get_metrics_dashboard()
        
        assert dashboard1 is dashboard2
    
    @patch("src.tasks.metrics.get_task_history")
    def test_get_dashboard_metrics_returns_dict(self, mock_history):
        """Test convenience function returns dictionary."""
        mock_history_instance = MagicMock()
        mock_history_instance.get_history.return_value = []
        mock_history.return_value = mock_history_instance
        
        result = get_dashboard_metrics(days=7)
        
        assert isinstance(result, dict)
        assert "summary" in result
        assert "task_metrics" in result

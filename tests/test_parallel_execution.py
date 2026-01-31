"""Tests for parallel task execution."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.task_registry import (
    run_tasks_parallel,
    run_category,
    list_task_categories,
    get_task_dependencies,
    TASK_CATEGORIES,
    TASK_DEPENDENCIES,
)


class TestTaskCategories:
    """Tests for task categorization."""

    def test_list_task_categories_returns_all_categories(self):
        categories = list_task_categories()
        assert "structure" in categories
        assert "quality" in categories
        assert "security" in categories
        assert "duplication" in categories
        assert "architecture" in categories
        assert "synthesis" in categories

    def test_structure_category_contents(self):
        assert "analyze-structure" in TASK_CATEGORIES["structure"]
        assert "tech-stack" in TASK_CATEGORIES["structure"]
        assert "detect-patterns" in TASK_CATEGORIES["structure"]

    def test_quality_category_contents(self):
        assert "detect-anti-patterns" in TASK_CATEGORIES["quality"]
        assert "health-score" in TASK_CATEGORIES["quality"]
        assert "test-mapping" in TASK_CATEGORIES["quality"]
        assert "identify-silent-failures" in TASK_CATEGORIES["quality"]

    def test_get_task_dependencies(self):
        # synthesis-roadmap has dependencies
        deps = get_task_dependencies("synthesis-roadmap")
        assert "detect-anti-patterns" in deps
        assert "health-score" in deps
        
        # Most tasks have no dependencies
        deps = get_task_dependencies("analyze-structure")
        assert len(deps) == 0


class TestRunTasksParallel:
    """Tests for run_tasks_parallel function."""

    @patch("src.task_registry.run_task")
    @patch("src.task_registry.TaskContext")
    def test_runs_multiple_tasks(self, mock_ctx_class, mock_run_task):
        """Test that multiple tasks execute and return results."""
        # Mock context
        mock_ctx = MagicMock()
        mock_ctx.get_files.return_value = ["file1.py", "file2.py"]
        mock_ctx_class.return_value = mock_ctx
        
        # Mock task returns
        mock_run_task.side_effect = [
            {"files": 10},
            {"languages": ["python"]},
        ]
        
        results = run_tasks_parallel(
            ["analyze-structure", "tech-stack"],
            source_path="src",
            max_workers=2,
        )
        
        assert "analyze-structure" in results
        assert "tech-stack" in results
        assert mock_run_task.call_count == 2

    @patch("src.task_registry.run_task")
    @patch("src.task_registry.TaskContext")
    def test_handles_task_errors(self, mock_ctx_class, mock_run_task):
        """Test that errors in individual tasks are captured."""
        mock_ctx = MagicMock()
        mock_ctx.get_files.return_value = []
        mock_ctx_class.return_value = mock_ctx
        
        # First task succeeds, second fails
        mock_run_task.side_effect = [
            {"files": 10},
            Exception("Task failed"),
        ]
        
        results = run_tasks_parallel(
            ["analyze-structure", "tech-stack"],
            source_path="src",
        )
        
        assert "analyze-structure" in results
        assert "tech-stack" in results
        # One result should have error
        assert any("error" in r for r in results.values())

    def test_validates_task_names(self):
        """Test that unknown task names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown task"):
            run_tasks_parallel(
                ["nonexistent-task"],
                source_path="src",
            )

    @patch("src.task_registry.run_task")
    @patch("src.task_registry.TaskContext")
    def test_calls_progress_callback(self, mock_ctx_class, mock_run_task):
        """Test that progress callback is invoked."""
        mock_ctx = MagicMock()
        mock_ctx.get_files.return_value = []
        mock_ctx_class.return_value = mock_ctx
        
        mock_run_task.return_value = {"result": "ok"}
        progress_events = []
        
        def progress_callback(event):
            progress_events.append(event)
        
        run_tasks_parallel(
            ["analyze-structure"],
            source_path="src",
            progress_callback=progress_callback,
        )
        
        # Should have started and completed events
        statuses = [e["status"] for e in progress_events]
        assert "started" in statuses
        assert "completed" in statuses


class TestRunCategory:
    """Tests for run_category function."""

    @patch("src.task_registry.run_tasks_parallel")
    def test_runs_all_tasks_in_category(self, mock_parallel):
        """Test that all tasks in a category are run."""
        mock_parallel.return_value = {
            "analyze-structure": {"files": 10},
            "tech-stack": {"languages": ["python"]},
            "detect-patterns": {"patterns": []},
        }
        
        results = run_category("structure", source_path="src")
        
        mock_parallel.assert_called_once()
        call_args = mock_parallel.call_args
        task_names = call_args[0][0] if call_args[0] else call_args[1]["task_names"]
        
        assert "analyze-structure" in task_names
        assert "tech-stack" in task_names
        assert "detect-patterns" in task_names

    def test_invalid_category_raises_error(self):
        """Test that invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Unknown category"):
            run_category("invalid-category", source_path="src")


class TestParallelExecutionPerformance:
    """Tests for parallel execution performance characteristics."""

    @patch("src.task_registry.run_task")
    @patch("src.task_registry.TaskContext")
    def test_tasks_share_file_cache(self, mock_ctx_class, mock_run_task):
        """Test that file cache is shared across parallel tasks."""
        mock_ctx = MagicMock()
        mock_ctx.get_files.return_value = ["file1.py"]
        mock_ctx_class.return_value = mock_ctx
        
        mock_run_task.return_value = {"result": "ok"}
        
        run_tasks_parallel(
            ["analyze-structure", "tech-stack"],
            source_path="src",
        )
        
        # Context should be created once
        mock_ctx_class.assert_called_once()
        # Files should be fetched once (pre-warm)
        mock_ctx.get_files.assert_called_once()

    @patch("src.task_registry.run_task")
    @patch("src.task_registry.TaskContext")
    def test_max_workers_respected(self, mock_ctx_class, mock_run_task):
        """Test that max_workers parameter limits concurrency."""
        mock_ctx = MagicMock()
        mock_ctx.get_files.return_value = []
        mock_ctx_class.return_value = mock_ctx
        
        # Slow task to test concurrency
        def slow_task(*args, **kwargs):
            time.sleep(0.1)
            return {"result": "ok"}
        
        mock_run_task.side_effect = slow_task
        
        start = time.time()
        run_tasks_parallel(
            ["analyze-structure", "tech-stack", "detect-patterns"],
            source_path="src",
            max_workers=1,  # Sequential
        )
        sequential_time = time.time() - start
        
        # With max_workers=1, tasks run sequentially
        # 3 tasks * 0.1s = ~0.3s minimum
        assert sequential_time >= 0.25

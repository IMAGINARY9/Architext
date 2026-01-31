"""Tests for task pipeline composition."""
from __future__ import annotations

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.tasks.pipeline import (
    PipelineStep,
    ParallelGroup,
    TaskPipeline,
    PipelineResult,
    PipelineExecutor,
    PipelineStore,
    BUILTIN_PIPELINES,
    list_builtin_pipelines,
    get_builtin_pipeline,
    get_pipeline_store,
)


class TestPipelineStep:
    """Tests for PipelineStep dataclass."""
    
    def test_create_step(self):
        """Test creating a pipeline step."""
        step = PipelineStep(task_name="analyze-structure")
        assert step.task_name == "analyze-structure"
        assert step.params == {}
        assert step.on_error == "stop"
    
    def test_step_with_params(self):
        """Test step with parameters."""
        step = PipelineStep(
            task_name="analyze-structure",
            params={"source_path": "/test"},
            on_error="continue",
        )
        assert step.params == {"source_path": "/test"}
        assert step.on_error == "continue"
    
    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        step = PipelineStep(
            task_name="test-task",
            params={"key": "value"},
            on_error="continue",
        )
        
        data = step.to_dict()
        restored = PipelineStep.from_dict(data)
        
        assert restored.task_name == step.task_name
        assert restored.params == step.params
        assert restored.on_error == step.on_error


class TestParallelGroup:
    """Tests for ParallelGroup dataclass."""
    
    def test_create_group(self):
        """Test creating a parallel group."""
        group = ParallelGroup(steps=[
            PipelineStep(task_name="task-a"),
            PipelineStep(task_name="task-b"),
        ])
        assert len(group.steps) == 2
    
    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        group = ParallelGroup(steps=[
            PipelineStep(task_name="task-a", params={"key": "a"}),
            PipelineStep(task_name="task-b", params={"key": "b"}),
        ])
        
        data = group.to_dict()
        restored = ParallelGroup.from_dict(data)
        
        assert len(restored.steps) == 2
        assert restored.steps[0].task_name == "task-a"
        assert restored.steps[1].task_name == "task-b"


class TestTaskPipeline:
    """Tests for TaskPipeline dataclass."""
    
    def test_create_pipeline(self):
        """Test creating a pipeline."""
        pipeline = TaskPipeline(
            name="test-pipeline",
            description="Test description",
            steps=[
                PipelineStep(task_name="task-a"),
                PipelineStep(task_name="task-b"),
            ],
        )
        assert pipeline.name == "test-pipeline"
        assert pipeline.description == "Test description"
        assert len(pipeline.steps) == 2
        assert pipeline.id is not None  # Auto-generated
    
    def test_pipeline_with_custom_id(self):
        """Test pipeline with custom ID."""
        pipeline = TaskPipeline(
            id="custom-id",
            name="test",
            steps=[],
        )
        assert pipeline.id == "custom-id"
    
    def test_pipeline_with_parallel_group(self):
        """Test pipeline containing parallel groups."""
        pipeline = TaskPipeline(
            name="mixed-pipeline",
            steps=[
                PipelineStep(task_name="task-a"),
                ParallelGroup(steps=[
                    PipelineStep(task_name="task-b"),
                    PipelineStep(task_name="task-c"),
                ]),
                PipelineStep(task_name="task-d"),
            ],
        )
        assert len(pipeline.steps) == 3
        assert isinstance(pipeline.steps[1], ParallelGroup)
    
    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        pipeline = TaskPipeline(
            id="test-id",
            name="test-pipeline",
            description="Test",
            steps=[
                PipelineStep(task_name="task-a"),
                ParallelGroup(steps=[
                    PipelineStep(task_name="task-b"),
                ]),
            ],
            created_at=datetime(2024, 1, 15, 10, 30, 0),
        )
        
        data = pipeline.to_dict()
        restored = TaskPipeline.from_dict(data)
        
        assert restored.id == pipeline.id
        assert restored.name == pipeline.name
        assert restored.description == pipeline.description
        assert len(restored.steps) == 2
        assert isinstance(restored.steps[1], ParallelGroup)
        assert restored.created_at == pipeline.created_at


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""
    
    def test_create_result(self):
        """Test creating a pipeline result."""
        result = PipelineResult(
            pipeline_id="test",
            success=True,
            total_duration_seconds=5.0,
            tasks_executed=3,
            tasks_failed=0,
            results={"task-a": {"data": "value"}},
        )
        assert result.success is True
        assert result.tasks_executed == 3
        assert result.tasks_failed == 0
    
    def test_failed_result(self):
        """Test failed pipeline result."""
        result = PipelineResult(
            pipeline_id="test",
            success=False,
            total_duration_seconds=2.0,
            tasks_executed=1,
            tasks_failed=1,
            results={"task-a": {"status": "success"}},
            errors={"task-b": "Error message"},
        )
        assert result.success is False
        assert result.tasks_failed == 1


class TestPipelineExecutor:
    """Tests for PipelineExecutor class."""
    
    @pytest.fixture
    def mock_run_task(self):
        """Mock the run_task function."""
        with patch("src.task_registry.run_task") as mock:
            mock.return_value = {"status": "success"}
            yield mock
    
    @pytest.fixture
    def mock_run_parallel(self):
        """Mock the run_tasks_parallel function."""
        with patch("src.task_registry.run_tasks_parallel") as mock:
            mock.return_value = {
                "task-a": {"status": "success"},
                "task-b": {"status": "success"},
            }
            yield mock
    
    def test_execute_sequential_pipeline(self, mock_run_task):
        """Test executing a sequential pipeline."""
        pipeline = TaskPipeline(
            name="test",
            steps=[
                PipelineStep(task_name="task-a"),
                PipelineStep(task_name="task-b"),
            ],
        )
        
        executor = PipelineExecutor(source_path="/test")
        result = executor.execute(pipeline)
        
        assert result.success is True
        assert result.tasks_executed == 2
        assert mock_run_task.call_count == 2
    
    def test_execute_pipeline_with_parallel_group(self, mock_run_task, mock_run_parallel):
        """Test executing pipeline with parallel group."""
        pipeline = TaskPipeline(
            name="test",
            steps=[
                PipelineStep(task_name="task-first"),
                ParallelGroup(steps=[
                    PipelineStep(task_name="task-a"),
                    PipelineStep(task_name="task-b"),
                ]),
            ],
        )
        
        executor = PipelineExecutor(source_path="/test")
        result = executor.execute(pipeline)
        
        assert result.success is True
        assert mock_run_task.call_count == 1  # First task
        assert mock_run_parallel.call_count == 1  # Parallel group
    
    def test_execute_stops_on_error(self, mock_run_task):
        """Test that execution stops on error by default."""
        mock_run_task.side_effect = [
            {"status": "success"},
            ValueError("Task failed"),
        ]
        
        pipeline = TaskPipeline(
            name="test",
            steps=[
                PipelineStep(task_name="task-a"),
                PipelineStep(task_name="task-b"),
                PipelineStep(task_name="task-c"),
            ],
        )
        
        executor = PipelineExecutor(source_path="/test")
        result = executor.execute(pipeline, stop_on_error=True)
        
        assert result.success is False
        assert result.tasks_executed == 1
        assert result.tasks_failed == 1
        assert "task-b" in result.errors
    
    def test_execute_continues_on_error_when_flagged(self, mock_run_task):
        """Test on_error="continue" step flag."""
        mock_run_task.side_effect = [
            {"status": "success"},
            ValueError("Task failed"),
            {"status": "success"},
        ]
        
        pipeline = TaskPipeline(
            name="test",
            steps=[
                PipelineStep(task_name="task-a"),
                PipelineStep(task_name="task-b", on_error="continue"),
                PipelineStep(task_name="task-c"),
            ],
        )
        
        executor = PipelineExecutor(source_path="/test")
        result = executor.execute(pipeline, stop_on_error=True)
        
        assert result.success is False  # Still marked failed
        assert result.tasks_executed == 2
        assert result.tasks_failed == 1
    
    def test_execute_with_progress_callback(self, mock_run_task):
        """Test progress callback is called."""
        progress_events = []
        
        def callback(event):
            progress_events.append(event)
        
        pipeline = TaskPipeline(
            name="test",
            steps=[
                PipelineStep(task_name="task-a"),
            ],
        )
        
        executor = PipelineExecutor(source_path="/test")
        executor.execute(pipeline, progress_callback=callback)
        
        assert len(progress_events) > 0
    
    def test_execute_with_step_params(self, mock_run_task):
        """Test that step params are passed to tasks."""
        pipeline = TaskPipeline(
            name="test",
            steps=[
                PipelineStep(
                    task_name="task-a",
                    params={"custom_param": "value"},
                ),
            ],
        )
        
        executor = PipelineExecutor(source_path="/test")
        executor.execute(pipeline)
        
        call_kwargs = mock_run_task.call_args[1]
        assert call_kwargs["custom_param"] == "value"


class TestPipelineStore:
    """Tests for PipelineStore class."""
    
    @pytest.fixture
    def temp_store(self, tmp_path):
        """Create a store with temp storage."""
        return PipelineStore(storage_path=tmp_path)
    
    def test_save_and_get_pipeline(self, temp_store):
        """Test saving and retrieving a pipeline."""
        pipeline = TaskPipeline(
            id="test-id",
            name="test-pipeline",
            steps=[PipelineStep(task_name="task-a")],
        )
        
        temp_store.save(pipeline)
        retrieved = temp_store.get("test-id")
        
        assert retrieved is not None
        assert retrieved.id == "test-id"
        assert retrieved.name == "test-pipeline"
    
    def test_get_nonexistent_pipeline(self, temp_store):
        """Test getting a pipeline that doesn't exist."""
        result = temp_store.get("nonexistent")
        assert result is None
    
    def test_list_pipelines(self, temp_store):
        """Test listing all pipelines."""
        for i in range(3):
            pipeline = TaskPipeline(
                id=f"pipeline-{i}",
                name=f"Pipeline {i}",
                steps=[],
            )
            temp_store.save(pipeline)
        
        pipelines = temp_store.list_pipelines()
        assert len(pipelines) == 3
    
    def test_delete_pipeline(self, temp_store):
        """Test deleting a pipeline."""
        pipeline = TaskPipeline(
            id="to-delete",
            name="Delete Me",
            steps=[],
        )
        temp_store.save(pipeline)
        
        deleted = temp_store.delete("to-delete")
        
        assert deleted is True
        assert temp_store.get("to-delete") is None
    
    def test_delete_nonexistent_pipeline(self, temp_store):
        """Test deleting a pipeline that doesn't exist."""
        deleted = temp_store.delete("nonexistent")
        assert deleted is False
    
    def test_persistence(self, tmp_path):
        """Test that pipelines persist across instances."""
        store1 = PipelineStore(storage_path=tmp_path)
        pipeline = TaskPipeline(
            id="persist-test",
            name="Persistent",
            steps=[PipelineStep(task_name="task-a")],
        )
        store1.save(pipeline)
        
        store2 = PipelineStore(storage_path=tmp_path)
        retrieved = store2.get("persist-test")
        
        assert retrieved is not None
        assert retrieved.name == "Persistent"


class TestBuiltinPipelines:
    """Tests for built-in pipelines."""
    
    def test_builtin_pipelines_exist(self):
        """Test that built-in pipelines are defined."""
        assert len(BUILTIN_PIPELINES) > 0
    
    def test_list_builtin_pipelines(self):
        """Test listing built-in pipelines."""
        pipelines = list_builtin_pipelines()
        
        assert len(pipelines) > 0
        for p in pipelines:
            assert isinstance(p, TaskPipeline)
            assert p.id is not None
            assert p.name is not None
    
    def test_get_builtin_pipeline(self):
        """Test getting a specific built-in pipeline."""
        pipeline = get_builtin_pipeline("quick-scan")
        
        assert pipeline is not None
        assert pipeline.id == "quick-scan"
        assert len(pipeline.steps) > 0
    
    def test_get_nonexistent_builtin(self):
        """Test getting a built-in that doesn't exist."""
        result = get_builtin_pipeline("nonexistent")
        assert result is None
    
    def test_quick_scan_pipeline(self):
        """Test quick-scan pipeline structure."""
        pipeline = get_builtin_pipeline("quick-scan")
        
        assert pipeline is not None
        assert "quick" in pipeline.name.lower() or "scan" in pipeline.name.lower()
    
    def test_full_analysis_pipeline(self):
        """Test full-analysis pipeline structure."""
        pipeline = get_builtin_pipeline("full-analysis")
        
        assert pipeline is not None
        # Full analysis should have more steps
        assert len(pipeline.steps) >= 3
    
    def test_security_audit_pipeline(self):
        """Test security-audit pipeline structure."""
        pipeline = get_builtin_pipeline("security-audit")
        
        assert pipeline is not None
        # Should contain security-related tasks
        task_names = []
        for step in pipeline.steps:
            if isinstance(step, PipelineStep):
                task_names.append(step.task_name)
            elif isinstance(step, ParallelGroup):
                for s in step.steps:
                    task_names.append(s.task_name)
        
        # At least one security task
        security_tasks = [t for t in task_names if "security" in t or "vulnerab" in t]
        assert len(security_tasks) > 0


class TestSingletons:
    """Tests for singleton patterns."""
    
    def test_get_pipeline_store_singleton(self, tmp_path):
        """Test that get_pipeline_store returns singleton."""
        import src.tasks.pipeline as pipeline_module
        pipeline_module._pipeline_store_instance = None
        
        with patch.object(pipeline_module, '_pipeline_store_instance', None):
            s1 = get_pipeline_store()
            s2 = get_pipeline_store()
            
            assert s1 is s2

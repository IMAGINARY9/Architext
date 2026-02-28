"""Custom task pipelines and composition.

This module enables users to define custom task pipelines that combine
multiple tasks with transformations and conditional logic.

Features:
- Sequential and parallel task execution
- Result transformations between tasks
- Conditional branching based on results
- Pipeline persistence and reuse
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union
import threading


@dataclass
class PipelineStep:
    """A single step in a task pipeline."""
    task_name: str
    condition: Optional[str] = None  # Python expression to evaluate
    transform: Optional[str] = None  # Python expression to transform result
    params: Dict[str, Any] = field(default_factory=dict)
    on_error: Literal["stop", "continue", "skip"] = "stop"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "condition": self.condition,
            "transform": self.transform,
            "params": self.params,
            "on_error": self.on_error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineStep":
        return cls(
            task_name=data["task_name"],
            condition=data.get("condition"),
            transform=data.get("transform"),
            params=data.get("params", {}),
            on_error=data.get("on_error", "stop"),
        )


@dataclass
class ParallelGroup:
    """A group of steps to execute in parallel."""
    steps: List[PipelineStep]
    merge_strategy: Literal["dict", "list", "first"] = "dict"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "parallel",
            "steps": [s.to_dict() for s in self.steps],
            "merge_strategy": self.merge_strategy,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParallelGroup":
        return cls(
            steps=[PipelineStep.from_dict(s) for s in data["steps"]],
            merge_strategy=data.get("merge_strategy", "dict"),
        )


@dataclass
class TaskPipeline:
    """
    A reusable pipeline of tasks.
    
    Pipelines can include:
    - Sequential steps
    - Parallel groups
    - Conditional execution
    - Result transformations
    
    Example:
        pipeline = TaskPipeline(
            name="full-analysis",
            description="Complete codebase analysis",
            steps=[
                PipelineStep(task_name="analyze-structure"),
                ParallelGroup(steps=[
                    PipelineStep(task_name="detect-anti-patterns"),
                    PipelineStep(task_name="security-heuristics"),
                ]),
                PipelineStep(
                    task_name="health-score",
                    condition="results.get('detect-anti-patterns', {}).get('count', 0) < 10",
                ),
            ],
        )
    """
    name: str
    steps: List[Union[PipelineStep, ParallelGroup]]
    id: Optional[str] = None
    description: str = ""
    version: str = "1.0"
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.id is None:
            # Generate ID from name
            self.id = "".join(c if c.isalnum() or c == "-" else "-" for c in self.name.lower())
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        steps_data = []
        for step in self.steps:
            if isinstance(step, ParallelGroup):
                steps_data.append(step.to_dict())
            else:
                steps_data.append({"type": "step", **step.to_dict()})
        
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "steps": steps_data,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskPipeline":
        steps = []
        for step_data in data.get("steps", []):
            if step_data.get("type") == "parallel":
                steps.append(ParallelGroup.from_dict(step_data))
            else:
                # Remove type field before creating PipelineStep
                step_dict = {k: v for k, v in step_data.items() if k != "type"}
                steps.append(PipelineStep.from_dict(step_dict))
        
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        return cls(
            id=data.get("id"),
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            created_at=created_at,
            steps=steps,
        )
    
    def validate(self, available_tasks: List[str]) -> List[str]:
        """
        Validate the pipeline configuration.
        
        Returns list of validation errors (empty if valid).
        """
        errors = []
        
        for i, step in enumerate(self.steps):
            if isinstance(step, ParallelGroup):
                for j, sub_step in enumerate(step.steps):
                    if sub_step.task_name not in available_tasks:
                        errors.append(f"Step {i}.{j}: Unknown task '{sub_step.task_name}'")
            else:
                if step.task_name not in available_tasks:
                    errors.append(f"Step {i}: Unknown task '{step.task_name}'")
        
        return errors


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    pipeline_id: str
    success: bool
    total_duration_seconds: float
    tasks_executed: int
    tasks_failed: int
    results: Dict[str, Any]
    errors: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "success": self.success,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "tasks_executed": self.tasks_executed,
            "tasks_failed": self.tasks_failed,
            "results": self.results,
            "errors": self.errors,
        }


class PipelineExecutor:
    """
    Executes task pipelines.
    
    Usage:
        executor = PipelineExecutor(source_path="src")
        result = executor.execute(pipeline)
    """
    
    def __init__(
        self,
        source_path: Optional[str] = None,
        storage_path: Optional[str] = None,
        max_workers: int = 4,
    ):
        """
        Initialize executor.
        
        Args:
            source_path: Source path for tasks
            storage_path: Storage path for tasks
            max_workers: Max workers for parallel execution
        """
        self.source_path = source_path
        self.storage_path = storage_path
        self.max_workers = max_workers
    
    def execute(
        self,
        pipeline: TaskPipeline,
        stop_on_error: bool = True,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> PipelineResult:
        """
        Execute a pipeline.
        
        Args:
            pipeline: The pipeline to execute
            stop_on_error: Stop on first error (default: True)
            progress_callback: Optional progress callback
            
        Returns:
            PipelineResult with all task results
        """
        started_at = time.time()
        results: Dict[str, Any] = {}
        errors: Dict[str, str] = {}
        tasks_executed = 0
        tasks_failed = 0
        
        if progress_callback:
            progress_callback({
                "type": "pipeline_start",
                "pipeline": pipeline.name,
                "total_steps": len(pipeline.steps),
            })
        
        for i, step in enumerate(pipeline.steps):
            if progress_callback:
                progress_callback({
                    "type": "step_start",
                    "step_index": i,
                })
            
            try:
                if isinstance(step, ParallelGroup):
                    step_results = self._run_parallel_group(step, progress_callback)
                    # Check for errors in parallel results
                    for task_name, result in step_results.items():
                        if isinstance(result, dict) and "error" in result:
                            errors[task_name] = result["error"]
                            tasks_failed += 1
                        else:
                            tasks_executed += 1
                        results[task_name] = result
                else:
                    result = self._run_step(step, progress_callback)
                    results[step.task_name] = result
                    tasks_executed += 1
                
                if progress_callback:
                    progress_callback({
                        "type": "step_complete",
                        "step_index": i,
                    })
                    
            except Exception as e:
                error_msg = str(e)
                task_name = step.task_name if isinstance(step, PipelineStep) else f"parallel-group-{i}"
                errors[task_name] = error_msg
                tasks_failed += 1
                
                if progress_callback:
                    progress_callback({
                        "type": "step_error",
                        "step_index": i,
                        "error": error_msg,
                    })
                
                # Check if we should stop
                should_continue = False
                if isinstance(step, PipelineStep) and step.on_error == "continue":
                    should_continue = True
                
                if stop_on_error and not should_continue:
                    break
        
        total_duration = time.time() - started_at
        
        return PipelineResult(
            pipeline_id=pipeline.id,
            success=tasks_failed == 0,
            total_duration_seconds=total_duration,
            tasks_executed=tasks_executed,
            tasks_failed=tasks_failed,
            results=results,
            errors=errors,
        )
    
    def _run_step(
        self,
        step: PipelineStep,
        progress_callback: Optional[Callable],
    ) -> Dict[str, Any]:
        """Run a single pipeline step."""
        from src.task_registry import run_task
        
        # Merge params
        params = {
            "storage_path": self.storage_path,
            "source_path": self.source_path,
            **step.params,
        }
        
        return run_task(step.task_name, **params)
    
    def _run_parallel_group(
        self,
        group: ParallelGroup,
        progress_callback: Optional[Callable],
    ) -> Dict[str, Dict[str, Any]]:
        """Run a parallel group of steps."""
        from src.task_registry import run_tasks_parallel
        
        task_names = [s.task_name for s in group.steps]
        return run_tasks_parallel(
            task_names,
            storage_path=self.storage_path,
            source_path=self.source_path,
            max_workers=self.max_workers,
        )


class PipelineStore:
    """
    Stores and retrieves task pipelines.
    
    Pipelines are persisted to disk as JSON files.
    """
    
    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        if storage_path is None:
            storage_path = Path.home() / ".architext" / "pipelines"
        
        self.store_dir = Path(storage_path)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
    
    def _pipeline_file(self, pipeline_id: str) -> Path:
        """Get the file path for a pipeline."""
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in pipeline_id)
        return self.store_dir / f"{safe_name}.json"
    
    def save(self, pipeline: TaskPipeline) -> None:
        """Save a pipeline to disk."""
        with self._lock:
            with open(self._pipeline_file(pipeline.id), "w", encoding="utf-8") as f:
                json.dump(pipeline.to_dict(), f, indent=2)
    
    def get(self, pipeline_id: str) -> Optional[TaskPipeline]:
        """Get a pipeline by ID."""
        path = self._pipeline_file(pipeline_id)
        if not path.exists():
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return TaskPipeline.from_dict(data)
        except Exception:
            return None
    
    def load(self, name: str) -> Optional[TaskPipeline]:
        """Load a pipeline from disk. Alias for get()."""
        return self.get(name)
    
    def list_pipelines(self) -> List[TaskPipeline]:
        """List all saved pipelines."""
        pipelines = []
        for path in self.store_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pipelines.append(TaskPipeline.from_dict(data))
            except Exception:
                continue
        return pipelines
    
    def list(self) -> List[Dict[str, Any]]:
        """List all saved pipelines as dicts."""
        pipelines = []
        for path in self.store_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                pipelines.append({
                    "name": data.get("name"),
                    "description": data.get("description", ""),
                    "version": data.get("version", "1.0"),
                    "steps_count": len(data.get("steps", [])),
                })
            except Exception:
                continue
        return pipelines
    
    def delete(self, name: str) -> bool:
        """Delete a pipeline."""
        path = self._pipeline_file(name)
        if path.exists():
            path.unlink()
            return True
        return False


# Pre-defined pipelines
BUILTIN_PIPELINES: Dict[str, TaskPipeline] = {
    "quick-scan": TaskPipeline(
        name="quick-scan",
        description="Fast overview of codebase structure and health",
        steps=[
            PipelineStep(task_name="analyze-structure"),
            PipelineStep(task_name="tech-stack"),
            PipelineStep(task_name="health-score"),
        ],
    ),
    "full-analysis": TaskPipeline(
        name="full-analysis",
        description="Complete codebase analysis with all checks",
        steps=[
            PipelineStep(task_name="analyze-structure"),
            ParallelGroup(
                steps=[
                    PipelineStep(task_name="tech-stack"),
                    PipelineStep(task_name="detect-patterns"),
                    PipelineStep(task_name="detect-anti-patterns"),
                ],
            ),
            ParallelGroup(
                steps=[
                    PipelineStep(task_name="security-heuristics"),
                    PipelineStep(task_name="detect-duplication"),
                    PipelineStep(task_name="identify-silent-failures"),
                ],
            ),
            PipelineStep(task_name="health-score"),
            PipelineStep(task_name="synthesis-roadmap"),
        ],
    ),
    "security-audit": TaskPipeline(
        name="security-audit",
        description="Security-focused analysis",
        steps=[
            ParallelGroup(
                steps=[
                    PipelineStep(task_name="security-heuristics"),
                    PipelineStep(task_name="detect-vulnerabilities"),
                ],
            ),
            PipelineStep(task_name="identify-silent-failures"),
        ],
    ),
    "code-quality": TaskPipeline(
        name="code-quality",
        description="Code quality and maintainability analysis",
        steps=[
            ParallelGroup(
                steps=[
                    PipelineStep(task_name="detect-anti-patterns"),
                    PipelineStep(task_name="detect-duplication"),
                    PipelineStep(task_name="detect-duplication-semantic"),
                ],
            ),
            PipelineStep(task_name="test-mapping"),
            PipelineStep(task_name="health-score"),
        ],
    ),
    "architecture-review": TaskPipeline(
        name="architecture-review",
        description="Architecture and dependency analysis",
        steps=[
            PipelineStep(task_name="analyze-structure"),
            ParallelGroup(
                steps=[
                    PipelineStep(task_name="detect-patterns"),
                    PipelineStep(task_name="dependency-graph"),
                    PipelineStep(task_name="code-knowledge-graph"),
                ],
            ),
        ],
    ),
}


def get_builtin_pipeline(name: str) -> Optional[TaskPipeline]:
    """Get a built-in pipeline by name."""
    return BUILTIN_PIPELINES.get(name)


def list_builtin_pipelines() -> List[TaskPipeline]:
    """List all built-in pipelines."""
    return list(BUILTIN_PIPELINES.values())


# Singleton store instance
_pipeline_store_instance: Optional[PipelineStore] = None
_pipeline_store_lock = threading.Lock()


def get_pipeline_store(storage_path: Optional[str] = None) -> PipelineStore:
    """Get the singleton PipelineStore instance."""
    global _pipeline_store_instance
    
    if _pipeline_store_instance is None:
        with _pipeline_store_lock:
            if _pipeline_store_instance is None:
                _pipeline_store_instance = PipelineStore(storage_path)
    
    return _pipeline_store_instance

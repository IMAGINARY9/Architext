"""Pipeline management endpoints.

Extracted from src/api/tasks.py — handles CRUD and execution of task pipelines.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException


def build_pipelines_router() -> APIRouter:
    """Create an APIRouter with all pipeline endpoints."""
    router = APIRouter()

    @router.get("/tasks/pipelines")
    async def list_pipelines() -> Dict[str, Any]:
        """List all available pipelines (built-in and custom)."""
        from src.tasks.orchestration.pipeline import list_builtin_pipelines, get_pipeline_store

        builtin = list_builtin_pipelines()
        store = get_pipeline_store()
        custom = store.list_pipelines()

        return {
            "builtin": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "step_count": len(p.steps),
                }
                for p in builtin
            ],
            "custom": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "step_count": len(p.steps),
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                }
                for p in custom
            ],
        }

    @router.get("/tasks/pipelines/{pipeline_id}")
    async def get_pipeline(pipeline_id: str) -> Dict[str, Any]:
        """Get details of a specific pipeline."""
        from src.tasks.orchestration.pipeline import get_builtin_pipeline, get_pipeline_store

        # Try built-in first
        pipeline = get_builtin_pipeline(pipeline_id)
        is_builtin = pipeline is not None

        # Try custom if not built-in
        if pipeline is None:
            store = get_pipeline_store()
            pipeline = store.get(pipeline_id)

        if pipeline is None:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline not found: {pipeline_id}"
            )

        def serialize_step(step):
            """Serialize a step (PipelineStep or ParallelGroup)."""
            from src.tasks.orchestration.pipeline import ParallelGroup
            if isinstance(step, ParallelGroup):
                return {
                    "type": "parallel",
                    "steps": [serialize_step(s) for s in step.steps],
                }
            return {
                "type": "task",
                "task_name": step.task_name,
                "params": step.params,
                "continue_on_error": step.continue_on_error,
            }

        return {
            "id": pipeline.id,
            "name": pipeline.name,
            "description": pipeline.description,
            "is_builtin": is_builtin,
            "steps": [serialize_step(s) for s in pipeline.steps],
            "created_at": pipeline.created_at.isoformat() if pipeline.created_at else None,
        }

    @router.post("/tasks/pipelines")
    async def create_pipeline(
        request: Dict[str, Any] = Body(
            ...,
            examples={
                "simple": {
                    "summary": "Simple sequential pipeline",
                    "value": {
                        "name": "my-scan",
                        "description": "Custom scan pipeline",
                        "steps": [
                            {"task_name": "analyze-structure"},
                            {"task_name": "tech-stack"},
                        ],
                    },
                },
                "with_parallel": {
                    "summary": "Pipeline with parallel steps",
                    "value": {
                        "name": "full-scan",
                        "description": "Comprehensive scan",
                        "steps": [
                            {"task_name": "analyze-structure"},
                            {
                                "parallel": [
                                    {"task_name": "detect-anti-patterns"},
                                    {"task_name": "detect-vulnerabilities"},
                                ]
                            },
                        ],
                    },
                },
            },
        ),
    ) -> Dict[str, Any]:
        """Create a custom pipeline.

        Steps can be either:
        - {"task_name": "...", "params": {...}, "continue_on_error": false}
        - {"parallel": [...steps...]} for parallel execution
        """
        from src.tasks.orchestration.pipeline import (
            TaskPipeline, PipelineStep, ParallelGroup, get_pipeline_store
        )

        name = request.get("name")
        if not name:
            raise HTTPException(status_code=400, detail="name is required")

        description = request.get("description", "")
        raw_steps = request.get("steps", [])

        if not raw_steps:
            raise HTTPException(status_code=400, detail="steps list is required")

        def parse_step(step_data):
            if "parallel" in step_data:
                return ParallelGroup(
                    steps=[parse_step(s) for s in step_data["parallel"]]
                )
            return PipelineStep(
                task_name=step_data["task_name"],
                params=step_data.get("params", {}),
                continue_on_error=step_data.get("continue_on_error", False),
            )

        try:
            steps = [parse_step(s) for s in raw_steps]
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Invalid step format: {e}")

        pipeline = TaskPipeline(
            name=name,
            description=description,
            steps=steps,
        )

        store = get_pipeline_store()
        store.save(pipeline)

        return {
            "status": "created",
            "id": pipeline.id,
            "name": pipeline.name,
        }

    @router.delete("/tasks/pipelines/{pipeline_id}")
    async def delete_pipeline(pipeline_id: str) -> Dict[str, Any]:
        """Delete a custom pipeline (built-in pipelines cannot be deleted)."""
        from src.tasks.orchestration.pipeline import get_builtin_pipeline, get_pipeline_store

        if get_builtin_pipeline(pipeline_id) is not None:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete built-in pipeline"
            )

        store = get_pipeline_store()
        deleted = store.delete(pipeline_id)

        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline not found: {pipeline_id}"
            )

        return {"status": "deleted", "id": pipeline_id}

    @router.post("/tasks/pipelines/{pipeline_id}/run")
    async def run_pipeline(
        pipeline_id: str,
        request: Dict[str, Any] = Body(
            default={},
            examples={
                "basic": {
                    "summary": "Run with source path",
                    "value": {"source": "./src"},
                },
            },
        ),
    ) -> Dict[str, Any]:
        """Execute a pipeline.

        Args:
            source: Source code path
            storage: ChromaDB storage path
            max_workers: Max parallel workers (default: 4)
            stop_on_error: Stop pipeline on first error (default: True)
        """
        from src.tasks.orchestration.pipeline import (
            get_builtin_pipeline, get_pipeline_store, PipelineExecutor
        )

        # Find pipeline
        pipeline = get_builtin_pipeline(pipeline_id)
        if pipeline is None:
            store = get_pipeline_store()
            pipeline = store.get(pipeline_id)

        if pipeline is None:
            raise HTTPException(
                status_code=404,
                detail=f"Pipeline not found: {pipeline_id}"
            )

        source_path = request.get("source")
        storage_path = request.get("storage")
        max_workers = request.get("max_workers", 4)
        stop_on_error = request.get("stop_on_error", True)

        if not source_path and not storage_path:
            raise HTTPException(
                status_code=400,
                detail="source or storage path is required"
            )

        executor = PipelineExecutor(
            source_path=source_path,
            storage_path=storage_path,
            max_workers=max_workers,
        )

        result = executor.execute(pipeline, stop_on_error=stop_on_error)

        return {
            "pipeline_id": pipeline_id,
            "pipeline_name": pipeline.name,
            "status": "completed" if result.success else "failed",
            "total_duration_seconds": result.total_duration_seconds,
            "tasks_executed": result.tasks_executed,
            "tasks_failed": result.tasks_failed,
            "results": result.results,
            "errors": result.errors,
        }

    return router

"""Task analysis endpoints."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type

from fastapi import APIRouter, Body, HTTPException

from src.task_registry import (
    list_task_categories,
    run_tasks_parallel,
    run_category as run_task_category,
    TASK_CATEGORIES,
)


SubmitTask = Callable[[str, Any], str]
RunTask = Callable[[str, Any], Dict[str, Any]]
UpdateTask = Callable[[str, Dict[str, Any]], Dict[str, Any]]


def build_tasks_router(
    submit_task: SubmitTask,
    run_task_inline: RunTask,
    update_task: UpdateTask,
    task_request_type: Type[Any],
    uuid_factory: Callable[[], str],
) -> APIRouter:
    router = APIRouter()

    def _parse_request(payload: Dict[str, Any]) -> Any:
        return task_request_type.model_validate(payload)

    def _inline_response(task_name: str, request: Any) -> Dict[str, Any]:
        task_id = uuid_factory()
        result = run_task_inline(task_name, request)
        update_task(task_id, {"status": "completed", "result": result, "task": task_name})
        return {"task_id": task_id, "status": "completed", "result": result}

    @router.post("/tasks/analyze-structure", status_code=202)
    async def analyze_structure_task(
        request: Dict[str, Any] = Body(
            ...,
            examples={
                "inline": {
                    "summary": "Run inline",
                    "value": {"source": "./src", "output_format": "json", "background": False},
                },
                "background": {
                    "summary": "Run as background task",
                    "value": {"source": "./src", "output_format": "json", "background": True},
                },
            },
        ),
    ) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("analyze-structure", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("analyze-structure", payload)

    @router.post("/tasks/tech-stack", status_code=202)
    async def tech_stack_task(
        request: Dict[str, Any] = Body(
            ...,
            examples={
                "inline": {"summary": "Inline tech stack", "value": {"source": "./src", "background": False}},
                "background": {"summary": "Background", "value": {"source": "./src", "background": True}},
            },
        ),
    ) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("tech-stack", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("tech-stack", payload)

    @router.post("/tasks/detect-anti-patterns", status_code=202)
    async def detect_anti_patterns_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("detect-anti-patterns", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("detect-anti-patterns", payload)

    @router.post("/tasks/health-score", status_code=202)
    async def health_score_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("health-score", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("health-score", payload)

    @router.post("/tasks/impact-analysis", status_code=202)
    async def impact_analysis_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("impact-analysis", payload)
            return {"task_id": task_id, "status": "queued"}
        if not payload.module:
            raise HTTPException(status_code=400, detail="module is required")
        return _inline_response("impact-analysis", payload)

    @router.post("/tasks/dependency-graph", status_code=202)
    async def dependency_graph_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("dependency-graph", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("dependency-graph", payload)

    @router.post("/tasks/test-mapping", status_code=202)
    async def test_mapping_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("test-mapping", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("test-mapping", payload)

    @router.post("/tasks/detect-patterns", status_code=202)
    async def detect_patterns_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("detect-patterns", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("detect-patterns", payload)

    @router.post("/tasks/detect-vulnerabilities", status_code=202)
    async def detect_vulnerabilities_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("detect-vulnerabilities", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("detect-vulnerabilities", payload)

    @router.post("/tasks/identify-silent-failures", status_code=202)
    async def identify_silent_failures_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("identify-silent-failures", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("identify-silent-failures", payload)

    @router.post("/tasks/security-heuristics", status_code=202)
    async def security_heuristics_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("security-heuristics", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("security-heuristics", payload)

    @router.post("/tasks/code-knowledge-graph", status_code=202)
    async def code_knowledge_graph_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("code-knowledge-graph", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("code-knowledge-graph", payload)

    @router.post("/tasks/synthesis-roadmap", status_code=202)
    async def synthesis_roadmap_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("synthesis-roadmap", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("synthesis-roadmap", payload)

    @router.get("/tasks/categories")
    async def get_task_categories() -> Dict[str, List[str]]:
        """Get all task categories with their task names."""
        return list_task_categories()

    @router.post("/tasks/run-parallel", status_code=200)
    async def run_parallel_tasks(
        request: Dict[str, Any] = Body(
            ...,
            examples={
                "multiple_tasks": {
                    "summary": "Run multiple tasks in parallel",
                    "value": {
                        "tasks": ["analyze-structure", "tech-stack", "detect-patterns"],
                        "source": "./src",
                        "max_workers": 4,
                    },
                },
            },
        ),
    ) -> Dict[str, Any]:
        """
        Run multiple tasks in parallel with shared file caching.
        
        This is more efficient than running tasks sequentially as file
        collection is done once and shared across all tasks.
        """
        task_names: List[str] = request.get("tasks", [])
        source_path: Optional[str] = request.get("source")
        storage_path: Optional[str] = request.get("storage")
        max_workers: int = request.get("max_workers", 4)
        
        if not task_names:
            raise HTTPException(status_code=400, detail="tasks list is required")
        if not source_path and not storage_path:
            raise HTTPException(status_code=400, detail="source or storage path is required")
        
        try:
            results = run_tasks_parallel(
                task_names=task_names,
                source_path=source_path,
                storage_path=storage_path,
                max_workers=max_workers,
            )
            return {
                "status": "completed",
                "tasks_run": len(task_names),
                "results": results,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/tasks/run-category/{category}", status_code=200)
    async def run_category_tasks(
        category: str,
        request: Dict[str, Any] = Body(
            ...,
            examples={
                "quality": {
                    "summary": "Run all quality tasks",
                    "value": {"source": "./src"},
                },
            },
        ),
    ) -> Dict[str, Any]:
        """
        Run all tasks in a category in parallel.
        
        Categories:
        - structure: analyze-structure, tech-stack, detect-patterns
        - quality: detect-anti-patterns, health-score, test-mapping, identify-silent-failures
        - security: detect-vulnerabilities, security-heuristics
        - duplication: detect-duplication, detect-duplication-semantic
        - architecture: impact-analysis, dependency-graph, code-knowledge-graph
        - synthesis: synthesis-roadmap
        """
        source_path: Optional[str] = request.get("source")
        storage_path: Optional[str] = request.get("storage")
        max_workers: int = request.get("max_workers", 4)
        
        if not source_path and not storage_path:
            raise HTTPException(status_code=400, detail="source or storage path is required")
        
        if category not in TASK_CATEGORIES:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown category: {category}. Valid: {list(TASK_CATEGORIES.keys())}"
            )
        
        try:
            results = run_task_category(
                category=category,
                source_path=source_path,
                storage_path=storage_path,
                max_workers=max_workers,
            )
            return {
                "status": "completed",
                "category": category,
                "tasks_run": len(TASK_CATEGORIES[category]),
                "results": results,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ===== Cache Management Endpoints =====
    
    @router.get("/tasks/cache/stats")
    async def get_cache_stats() -> Dict[str, Any]:
        """Get task result cache statistics."""
        from src.tasks.cache import get_task_cache
        cache = get_task_cache()
        return cache.get_stats()
    
    @router.post("/tasks/cache/clear")
    async def clear_cache(
        request: Dict[str, Any] = Body(
            default={},
            examples={
                "clear_all": {"summary": "Clear all cache", "value": {}},
                "clear_task": {"summary": "Clear specific task", "value": {"task_name": "analyze-structure"}},
            },
        ),
    ) -> Dict[str, Any]:
        """
        Clear task result cache.
        
        Optionally specify task_name to clear only that task's cache.
        """
        from src.tasks.cache import get_task_cache
        cache = get_task_cache()
        
        task_name = request.get("task_name")
        count = cache.invalidate(task_name=task_name)
        
        return {
            "status": "cleared",
            "entries_removed": count,
            "task_name": task_name or "all",
        }

    # ===== Execution History Endpoints =====
    
    @router.get("/tasks/history")
    async def get_execution_history(
        task_name: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Get task execution history.
        
        Args:
            task_name: Filter by specific task name
            limit: Maximum number of entries to return (default: 100)
        """
        from src.tasks.history import get_task_history
        history = get_task_history()
        
        executions = history.get_history(
            task_name=task_name,
            limit=limit,
        )
        
        return {
            "count": len(executions),
            "task_filter": task_name,
            "executions": [e.to_dict() for e in executions],
        }
    
    @router.get("/tasks/history/analytics")
    async def get_execution_analytics(
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get execution analytics for tasks.
        
        Returns statistics like average duration, success rate, etc.
        """
        from src.tasks.history import get_task_history
        history = get_task_history()
        
        analytics = history.get_analytics(task_name=task_name)
        
        if task_name and task_name not in analytics:
            raise HTTPException(
                status_code=404,
                detail=f"No history found for task: {task_name}"
            )
        
        return {
            "task_count": len(analytics),
            "tasks": {name: a.to_dict() for name, a in analytics.items()},
        }
    
    @router.delete("/tasks/history")
    async def clear_execution_history(
        task_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Clear execution history.
        
        Args:
            task_name: Clear history for specific task only
        """
        from src.tasks.history import get_task_history
        history = get_task_history()
        
        count = history.clear(task_name=task_name)
        
        return {
            "status": "cleared",
            "entries_removed": count,
            "task_name": task_name or "all",
        }
    
    # ===== Pipeline Endpoints =====
    
    @router.get("/tasks/pipelines")
    async def list_pipelines() -> Dict[str, Any]:
        """List all available pipelines (built-in and custom)."""
        from src.tasks.pipeline import list_builtin_pipelines, get_pipeline_store
        
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
        from src.tasks.pipeline import get_builtin_pipeline, get_pipeline_store
        
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
            from src.tasks.pipeline import ParallelGroup
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
        """
        Create a custom pipeline.
        
        Steps can be either:
        - {"task_name": "...", "params": {...}, "continue_on_error": false}
        - {"parallel": [...steps...]} for parallel execution
        """
        from src.tasks.pipeline import (
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
        from src.tasks.pipeline import get_builtin_pipeline, get_pipeline_store
        
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
        """
        Execute a pipeline.
        
        Args:
            source: Source code path
            storage: ChromaDB storage path
            max_workers: Max parallel workers (default: 4)
            stop_on_error: Stop pipeline on first error (default: True)
        """
        from src.tasks.pipeline import (
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

    # =========================================================================
    # Task Recommendations
    # =========================================================================
    
    @router.get("/tasks/recommendations")
    async def get_recommendations(
        limit: int = 5,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get task recommendations based on execution history.
        
        Args:
            limit: Maximum number of recommendations (default: 5)
            category: Prefer tasks from this category
        """
        from src.tasks.recommendations import get_task_recommendations
        
        recommendations = get_task_recommendations(
            limit=limit,
            prefer_category=category,
        )
        
        return {
            "recommendations": recommendations,
            "count": len(recommendations),
        }
    
    @router.get("/tasks/recommendations/quick-scan")
    async def get_quick_scan_recommendations() -> Dict[str, Any]:
        """Get recommendations for a quick codebase scan."""
        from src.tasks.recommendations import get_recommendation_engine
        
        engine = get_recommendation_engine()
        recommendations = engine.get_quick_scan_recommendation()
        
        return {
            "recommendations": [r.to_dict() for r in recommendations],
            "description": "Essential tasks for a quick codebase overview",
        }
    
    @router.get("/tasks/recommendations/category/{category}")
    async def get_category_recommendations(
        category: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Get recommendations for tasks in a specific category."""
        from src.tasks.recommendations import get_recommendation_engine
        
        if category not in TASK_CATEGORIES:
            raise HTTPException(
                status_code=404,
                detail=f"Category not found: {category}. Available: {list(TASK_CATEGORIES.keys())}"
            )
        
        engine = get_recommendation_engine()
        recommendations = engine.get_category_recommendations(category, limit)
        
        return {
            "category": category,
            "recommendations": [r.to_dict() for r in recommendations],
        }
    
    @router.get("/tasks/recommendations/related/{task_name}")
    async def get_related_recommendations(
        task_name: str,
        limit: int = 3,
    ) -> Dict[str, Any]:
        """Get recommendations for tasks related to a given task."""
        from src.tasks.recommendations import get_recommendation_engine
        from src.task_registry import TASK_REGISTRY
        
        if task_name not in TASK_REGISTRY:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found: {task_name}"
            )
        
        engine = get_recommendation_engine()
        recommendations = engine.get_related_recommendations(task_name, limit)
        
        return {
            "task_name": task_name,
            "related_recommendations": [r.to_dict() for r in recommendations],
        }

    # =========================================================================
    # Scoring Weights Configuration
    # =========================================================================
    
    @router.get("/tasks/recommendations/weights")
    async def get_scoring_weights() -> Dict[str, Any]:
        """Get current scoring weights for task recommendations."""
        from src.tasks.recommendations import get_scoring_weights as get_weights
        
        return {
            "weights": get_weights(),
            "description": "Current scoring weights used for task recommendations",
        }
    
    @router.put("/tasks/recommendations/weights")
    async def update_scoring_weights(
        request: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        """
        Update scoring weights for task recommendations.
        
        Body should contain weight names and values, e.g.:
        {"never_run_boost": 40.0, "stale_boost": 25.0}
        
        Available weights:
        - never_run_boost: Boost for tasks never executed (default: 30.0)
        - stale_boost: Boost for tasks not run recently (default: 20.0)
        - very_stale_boost: Additional boost for very old tasks (default: 25.0)
        - high_success_rate_boost: Boost for reliable tasks (default: 10.0)
        - low_success_rate_penalty: Penalty for failing tasks (default: -15.0)
        - fast_execution_boost: Boost for quick tasks (default: 5.0)
        - category_coverage_boost: Boost for underrepresented categories (default: 15.0)
        - dependency_boost: Boost for tasks with dependencies ready (default: 10.0)
        """
        from src.tasks.recommendations import update_scoring_weights as update_weights
        
        # Filter to only float values
        weight_updates = {k: float(v) for k, v in request.items() if isinstance(v, (int, float))}
        
        if not weight_updates:
            raise HTTPException(
                status_code=400,
                detail="No valid weight updates provided"
            )
        
        updated = update_weights(**weight_updates)
        
        return {
            "weights": updated,
            "updated_fields": list(weight_updates.keys()),
        }
    
    @router.post("/tasks/recommendations/weights/reset")
    async def reset_scoring_weights() -> Dict[str, Any]:
        """Reset scoring weights to default values."""
        from src.tasks.recommendations import reset_scoring_weights as reset_weights
        
        defaults = reset_weights()
        
        return {
            "weights": defaults,
            "message": "Scoring weights reset to defaults",
        }
    
    @router.get("/tasks/recommendations/weights/presets")
    async def get_weight_presets() -> Dict[str, Any]:
        """
        Get available scoring weight presets.
        
        Presets provide pre-configured weight combinations for different use cases:
        - default: Balanced recommendations
        - aggressive: Prioritize coverage and freshness
        - conservative: Prioritize reliability and speed
        - reliability-focused: Strongly prefer tasks that succeed
        - coverage-focused: Strongly prefer running all tasks
        """
        from src.tasks.recommendations import get_weight_presets as get_presets
        
        presets = get_presets()
        
        return {
            "presets": presets,
            "available": list(presets.keys()),
        }
    
    @router.post("/tasks/recommendations/weights/presets/{preset_name}")
    async def apply_weight_preset(preset_name: str) -> Dict[str, Any]:
        """
        Apply a scoring weight preset.
        
        Available presets: default, aggressive, conservative, 
        reliability-focused, coverage-focused
        """
        from src.tasks.recommendations import apply_weight_preset as apply_preset
        
        result = apply_preset(preset_name)
        
        if result is None:
            from src.tasks.recommendations import get_weight_presets as get_presets
            available = list(get_presets().keys())
            raise HTTPException(
                status_code=404,
                detail=f"Preset not found: {preset_name}. Available: {available}"
            )
        
        return {
            "preset": preset_name,
            "weights": result,
            "message": f"Applied preset: {preset_name}",
        }

    # =========================================================================
    # Metrics Dashboard
    # =========================================================================
    
    @router.get("/tasks/metrics/dashboard")
    async def get_dashboard(days: int = 30) -> Dict[str, Any]:
        """
        Get the complete metrics dashboard.
        
        Args:
            days: Number of days to include in analysis (default: 30)
        """
        from src.tasks.metrics import get_dashboard_metrics
        
        return get_dashboard_metrics(days)
    
    @router.get("/tasks/metrics/task/{task_name}")
    async def get_task_metrics(
        task_name: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get detailed metrics for a specific task.
        
        Args:
            task_name: Name of the task
            days: Number of days to include (default: 30)
        """
        from src.tasks.metrics import get_metrics_dashboard
        from src.task_registry import TASK_REGISTRY
        
        if task_name not in TASK_REGISTRY:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found: {task_name}"
            )
        
        dashboard = get_metrics_dashboard()
        return dashboard.get_task_details(task_name, days)
    
    @router.get("/tasks/metrics/summary")
    async def get_metrics_summary(days: int = 30) -> Dict[str, Any]:
        """
        Get a summary of execution metrics.
        
        Returns only the summary portion of the dashboard for quick overview.
        """
        from src.tasks.metrics import get_dashboard_metrics
        
        metrics = get_dashboard_metrics(days)
        return {
            "summary": metrics["summary"],
            "top_performers": metrics["top_performers"],
            "health": metrics["health"],
            "generated_at": metrics["generated_at"],
        }
    
    @router.get("/tasks/metrics/trends")
    async def get_execution_trends(days: int = 30) -> Dict[str, Any]:
        """
        Get daily execution trends.
        
        Returns execution counts and success rates by day.
        """
        from src.tasks.metrics import get_dashboard_metrics
        
        metrics = get_dashboard_metrics(days)
        return {
            "daily_trends": metrics["daily_trends"],
            "days": days,
        }
    
    @router.get("/tasks/metrics/categories")
    async def get_category_metrics(days: int = 30) -> Dict[str, Any]:
        """
        Get metrics grouped by category.
        
        Shows execution stats for each task category.
        """
        from src.tasks.metrics import get_dashboard_metrics
        
        metrics = get_dashboard_metrics(days)
        return {
            "category_metrics": metrics["category_metrics"],
            "days": days,
        }

    # =========================================================================
    # Webhooks
    # =========================================================================
    
    @router.get("/tasks/webhooks")
    async def list_webhooks() -> Dict[str, Any]:
        """List all registered webhooks."""
        from src.tasks.webhooks import get_webhook_manager
        
        manager = get_webhook_manager()
        webhooks = manager.list_webhooks()
        
        return {
            "webhooks": [w.to_dict() for w in webhooks],
            "count": len(webhooks),
        }
    
    @router.post("/tasks/webhooks")
    async def register_webhook(
        request: Dict[str, Any] = Body(
            ...,
            examples={
                "all_events": {
                    "summary": "Subscribe to all events",
                    "value": {
                        "url": "https://example.com/webhook",
                        "events": ["task.started", "task.completed", "task.failed"],
                    },
                },
                "with_secret": {
                    "summary": "With HMAC signature",
                    "value": {
                        "url": "https://example.com/webhook",
                        "events": ["task.completed"],
                        "secret": "my-secret-key",
                    },
                },
            },
        ),
    ) -> Dict[str, Any]:
        """
        Register a new webhook.
        
        Args:
            url: The URL to send webhook notifications to
            events: List of events to subscribe to
            secret: Optional secret for HMAC signature verification
            max_retries: Maximum retry attempts (default: 3)
            enabled: Whether webhook is active (default: True)
        """
        from src.tasks.webhooks import WebhookConfig, WebhookEvent, get_webhook_manager
        
        url = request.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="url is required")
        
        event_names = request.get("events", [])
        if not event_names:
            raise HTTPException(status_code=400, detail="events list is required")
        
        # Convert event names to WebhookEvent enum
        try:
            events = [WebhookEvent(name) for name in event_names]
        except ValueError as e:
            valid_events = [e.value for e in WebhookEvent]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event name. Valid events: {valid_events}"
            )
        
        config = WebhookConfig(
            url=url,
            events=events,
            secret=request.get("secret"),
            max_retries=request.get("max_retries", 3),
            enabled=request.get("enabled", True),
        )
        
        manager = get_webhook_manager()
        manager.register_webhook(config)
        
        return {
            "status": "registered",
            "webhook_id": config.id,
            "url": config.url,
            "events": [e.value for e in config.events],
        }
    
    @router.get("/tasks/webhooks/{webhook_id}")
    async def get_webhook(webhook_id: str) -> Dict[str, Any]:
        """Get details of a specific webhook."""
        from src.tasks.webhooks import get_webhook_manager
        
        manager = get_webhook_manager()
        webhook = manager.get_webhook(webhook_id)
        
        if webhook is None:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook not found: {webhook_id}"
            )
        
        return webhook.to_dict()
    
    @router.put("/tasks/webhooks/{webhook_id}")
    async def update_webhook(
        webhook_id: str,
        request: Dict[str, Any] = Body(...),
    ) -> Dict[str, Any]:
        """
        Update a webhook configuration.
        
        Args:
            url: New URL (optional)
            events: New events list (optional)
            secret: New secret (optional)
            enabled: Enable/disable (optional)
        """
        from src.tasks.webhooks import WebhookEvent, get_webhook_manager
        
        manager = get_webhook_manager()
        
        updates = {}
        if "url" in request:
            updates["url"] = request["url"]
        if "events" in request:
            try:
                updates["events"] = [WebhookEvent(name) for name in request["events"]]
            except ValueError:
                valid_events = [e.value for e in WebhookEvent]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid event. Valid: {valid_events}"
                )
        if "secret" in request:
            updates["secret"] = request["secret"]
        if "enabled" in request:
            updates["enabled"] = request["enabled"]
        if "max_retries" in request:
            updates["max_retries"] = request["max_retries"]
        
        webhook = manager.update_webhook(webhook_id, updates)
        
        if webhook is None:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook not found: {webhook_id}"
            )
        
        return {
            "status": "updated",
            "webhook": webhook.to_dict(),
        }
    
    @router.delete("/tasks/webhooks/{webhook_id}")
    async def delete_webhook(webhook_id: str) -> Dict[str, Any]:
        """Delete a webhook."""
        from src.tasks.webhooks import get_webhook_manager
        
        manager = get_webhook_manager()
        deleted = manager.unregister_webhook(webhook_id)
        
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook not found: {webhook_id}"
            )
        
        return {"status": "deleted", "webhook_id": webhook_id}
    
    @router.get("/tasks/webhooks/{webhook_id}/deliveries")
    async def get_webhook_deliveries(
        webhook_id: str,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get delivery history for a webhook."""
        from src.tasks.webhooks import get_webhook_manager
        
        manager = get_webhook_manager()
        webhook = manager.get_webhook(webhook_id)
        
        if webhook is None:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook not found: {webhook_id}"
            )
        
        deliveries = manager.get_deliveries(webhook_id, limit)
        
        return {
            "webhook_id": webhook_id,
            "deliveries": [d.to_dict() for d in deliveries],
            "count": len(deliveries),
        }
    
    @router.post("/tasks/webhooks/{webhook_id}/test")
    async def test_webhook(webhook_id: str) -> Dict[str, Any]:
        """Send a test event to a webhook."""
        from src.tasks.webhooks import WebhookEvent, get_webhook_manager
        
        manager = get_webhook_manager()
        webhook = manager.get_webhook(webhook_id)
        
        if webhook is None:
            raise HTTPException(
                status_code=404,
                detail=f"Webhook not found: {webhook_id}"
            )
        
        # Emit a test event
        delivery = manager.emit(
            event=WebhookEvent.TASK_COMPLETED,
            data={
                "task_name": "test-webhook",
                "task_id": "test-123",
                "status": "success",
                "duration_seconds": 0.1,
                "is_test": True,
            },
            webhook_ids=[webhook_id],
            async_delivery=False,  # Wait for result
        )
        
        if delivery:
            return {
                "status": "sent",
                "delivery": delivery[0].to_dict() if delivery else None,
            }
        return {"status": "no_delivery", "reason": "Webhook may not subscribe to task.completed"}

    # =========================================================================
    # Task Scheduling
    # =========================================================================
    
    @router.get("/tasks/schedules")
    async def list_schedules() -> Dict[str, Any]:
        """List all scheduled tasks."""
        from src.tasks.scheduler import get_task_scheduler
        
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
            examples={
                "interval": {
                    "summary": "Run every 30 minutes",
                    "value": {
                        "task_name": "health-score",
                        "schedule_type": "interval",
                        "interval_minutes": 30,
                        "source_path": "./src",
                    },
                },
                "cron": {
                    "summary": "Daily at 2am",
                    "value": {
                        "task_name": "detect-anti-patterns",
                        "schedule_type": "cron",
                        "cron_minute": "0",
                        "cron_hour": "2",
                        "source_path": "./src",
                    },
                },
            },
        ),
    ) -> Dict[str, Any]:
        """
        Create a scheduled task.
        
        Schedule types:
        - interval: Run every N minutes (specify interval_minutes)
        - cron: Cron-like schedule (specify cron_minute, cron_hour, cron_day_of_week)
        - once: Run once at specified time (specify run_at as ISO datetime)
        """
        from src.tasks.scheduler import (
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
        from src.tasks.scheduler import get_task_scheduler
        
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
        from src.tasks.scheduler import get_task_scheduler
        
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
        from src.tasks.scheduler import get_task_scheduler
        
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
        from src.tasks.scheduler import get_task_scheduler
        
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
        from src.tasks.scheduler import get_task_scheduler
        
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
        from src.tasks.scheduler import get_task_scheduler
        
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
        from src.tasks.scheduler import get_task_scheduler
        
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
        from src.tasks.scheduler import get_task_scheduler
        
        scheduler = get_task_scheduler()
        scheduler.start()
        
        return {
            "status": "started",
            "message": "Task scheduler is now running in background",
        }
    
    @router.post("/tasks/scheduler/stop")
    async def stop_scheduler() -> Dict[str, Any]:
        """Stop the background task scheduler."""
        from src.tasks.scheduler import get_task_scheduler
        
        scheduler = get_task_scheduler()
        scheduler.stop()
        
        return {
            "status": "stopped",
            "message": "Task scheduler has been stopped",
        }

    return router

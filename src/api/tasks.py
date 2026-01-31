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

    return router

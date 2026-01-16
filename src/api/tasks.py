"""Task analysis endpoints."""
from __future__ import annotations

from typing import Any, Callable, Dict, Type

from fastapi import APIRouter, Body, HTTPException


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
    async def analyze_structure_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("analyze-structure", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("analyze-structure", payload)

    @router.post("/tasks/tech-stack", status_code=202)
    async def tech_stack_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
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

    @router.post("/tasks/refactoring-recommendations", status_code=202)
    async def refactoring_recommendations_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("refactoring-recommendations", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("refactoring-recommendations", payload)

    @router.post("/tasks/generate-docs", status_code=202)
    async def generate_docs_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("generate-docs", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("generate-docs", payload)

    @router.post("/tasks/dependency-graph", status_code=202)
    async def dependency_graph_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("dependency-graph", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("dependency-graph", payload)

    @router.post("/tasks/test-coverage", status_code=202)
    async def test_coverage_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("test-coverage", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("test-coverage", payload)

    @router.post("/tasks/detect-patterns", status_code=202)
    async def detect_patterns_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("detect-patterns", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("detect-patterns", payload)

    @router.post("/tasks/diff-architecture", status_code=202)
    async def diff_architecture_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("diff-architecture", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("diff-architecture", payload)

    @router.post("/tasks/onboarding-guide", status_code=202)
    async def onboarding_guide_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("onboarding-guide", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("onboarding-guide", payload)

    @router.post("/tasks/detect-vulnerabilities", status_code=202)
    async def detect_vulnerabilities_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("detect-vulnerabilities", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("detect-vulnerabilities", payload)

    @router.post("/tasks/logic-gap-analysis", status_code=202)
    async def logic_gap_analysis_task(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        payload = _parse_request(request)
        if payload.background:
            task_id = submit_task("logic-gap-analysis", payload)
            return {"task_id": task_id, "status": "queued"}
        return _inline_response("logic-gap-analysis", payload)

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

    return router

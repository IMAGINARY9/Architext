"""FastAPI server exposing Architext indexing and query capabilities."""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, List
from uuid import uuid4

from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import ArchitextSettings, load_settings
from src.indexer import (
    create_index_from_paths,
    gather_index_files,
    initialize_settings,
    load_existing_index,
    query_index,
)
from src.ingestor import resolve_source, CACHE_DIR
from src.cli_utils import extract_sources, to_agent_response, to_agent_response_compact
from src.tasks import (
    analyze_structure,
    tech_stack,
    detect_anti_patterns,
    health_score,
    impact_analysis,
    refactoring_recommendations,
    generate_docs,
    dependency_graph_export,
    test_coverage_analysis,
    architecture_pattern_detection,
    diff_architecture_review,
    onboarding_guide,
    detect_vulnerabilities,
    logic_gap_analysis,
    identify_silent_failures,
    security_heuristics,
    code_knowledge_graph,
    synthesis_roadmap,
)


class IndexRequest(BaseModel):
    """Request payload to start indexing a repository."""

    source: str
    storage: Optional[str] = None
    no_cache: bool = False
    ssh_key: Optional[str] = None
    llm_provider: Optional[str] = Field(default=None, pattern="^(openai|local)$")
    embedding_provider: Optional[str] = Field(
        default=None, pattern="^(huggingface|openai)$"
    )
    background: bool = True
    dry_run: bool = False


class QueryRequest(BaseModel):
    """Request payload to query an existing index."""

    text: str
    storage: Optional[str] = None
    mode: str = Field(default="human", pattern="^(human|agent)$")
    enable_hybrid: Optional[bool] = None
    hybrid_alpha: Optional[float] = None
    enable_rerank: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_top_n: Optional[int] = None
    compact: Optional[bool] = None


class AskRequest(BaseModel):
    text: str
    storage: Optional[str] = None
    compact: bool = True
    enable_hybrid: Optional[bool] = None
    hybrid_alpha: Optional[float] = None
    enable_rerank: Optional[bool] = None
    rerank_model: Optional[str] = None
    rerank_top_n: Optional[int] = None


class TaskRequest(BaseModel):
    """Request payload for analysis tasks."""

    storage: Optional[str] = None
    source: Optional[str] = None
    output_format: str = Field(default="json", pattern="^(json|markdown|mermaid)$")
    depth: Optional[str] = Field(default="shallow", pattern="^(shallow|detailed|exhaustive)$")
    module: Optional[str] = None
    output_dir: Optional[str] = None
    background: bool = True


class MCPRunRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class _RateLimiter:
    """Simple in-memory token bucket per client IP."""

    def __init__(self, rate_per_minute: int, burst: Optional[int] = None):
        self.rate_per_minute = max(rate_per_minute, 1)
        self.capacity = burst or self.rate_per_minute
        self.tokens: Dict[str, float] = {}
        self.last_refill: Dict[str, float] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            tokens = self.tokens.get(key, self.capacity)
            last = self.last_refill.get(key, now)
            elapsed = max(0.0, now - last)
            refill = elapsed * (self.rate_per_minute / 60.0)
            tokens = min(self.capacity, tokens + refill)
            if tokens < 1.0:
                self.tokens[key] = tokens
                self.last_refill[key] = now
                return False
            self.tokens[key] = tokens - 1.0
            self.last_refill[key] = now
            return True


def _is_within_any(candidate: Path, roots: List[Path]) -> bool:
    for root in roots:
        try:
            candidate.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _parse_allowed_roots(raw: Optional[str], defaults: List[Path]) -> List[Path]:
    if raw:
        roots = [
            Path(item).expanduser().resolve()
            for item in raw.split(os.pathsep)
            if item.strip()
        ]
        return roots
    return defaults


def _resolve_task_store_path(raw_path: Optional[str]) -> Path:
    candidate = Path(raw_path or "~/.architext/task_store.json").expanduser().resolve()
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _sanitize_task_store(store: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    sanitized: Dict[str, Dict[str, Any]] = {}
    for task_id, payload in store.items():
        data = {}
        for key, value in payload.items():
            if key == "future":
                continue
            data[key] = value
        sanitized[task_id] = data
    return sanitized


def _mcp_tools_schema() -> List[Dict[str, Any]]:
    return [
        {
            "name": "architext.query",
            "description": "Query an index and return agent/human output.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "storage": {"type": "string"},
                    "mode": {"type": "string", "enum": ["human", "agent"]},
                    "compact": {"type": "boolean"},
                    "enable_hybrid": {"type": "boolean"},
                    "hybrid_alpha": {"type": "number"},
                    "enable_rerank": {"type": "boolean"},
                    "rerank_model": {"type": "string"},
                    "rerank_top_n": {"type": "integer"},
                },
                "required": ["text"],
            },
        },
        {
            "name": "architext.ask",
            "description": "Agent-optimized query with compact response support.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "storage": {"type": "string"},
                    "compact": {"type": "boolean"},
                    "enable_hybrid": {"type": "boolean"},
                    "hybrid_alpha": {"type": "number"},
                    "enable_rerank": {"type": "boolean"},
                    "rerank_model": {"type": "string"},
                    "rerank_top_n": {"type": "integer"},
                },
                "required": ["text"],
            },
        },
        {
            "name": "architext.task",
            "description": "Run an analysis task synchronously.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "storage": {"type": "string"},
                    "source": {"type": "string"},
                    "output_format": {"type": "string"},
                    "depth": {"type": "string"},
                    "module": {"type": "string"},
                    "output_dir": {"type": "string"},
                },
                "required": ["task"],
            },
        },
    ]


def _load_task_store(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    for payload in data.values():
        if payload.get("status") in {"queued", "running"}:
            payload["status"] = "stale"
            payload["note"] = "Server restarted before completion"
    return data


def _persist_task_store(path: Path, store: Dict[str, Dict[str, Any]]) -> None:
    try:
        sanitized = _sanitize_task_store(store)
        path.write_text(json.dumps(sanitized, indent=2, default=str), encoding="utf-8")
    except Exception:
        return


def _resolve_task_source(raw_path: Optional[str], source_roots: List[Path]) -> Optional[str]:
    if not raw_path:
        return None
    candidate = Path(raw_path).expanduser().resolve()
    if not candidate.exists():
        raise HTTPException(status_code=400, detail="source path does not exist")
    if not candidate.is_dir():
        raise HTTPException(status_code=400, detail="source path must be a directory")
    if not _is_within_any(candidate, source_roots):
        raise HTTPException(status_code=400, detail="source path must be within allowed roots")
    return str(candidate)


def create_app(settings: Optional[ArchitextSettings] = None) -> FastAPI:
    """Create a FastAPI app configured with Architext settings."""

    base_settings = settings or load_settings()
    initialize_settings(base_settings)

    task_store_path = _resolve_task_store_path(getattr(base_settings, "task_store_path", None))
    task_store: Dict[str, Dict[str, Any]] = _load_task_store(task_store_path)
    lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=2)

    storage_roots = _parse_allowed_roots(
        getattr(base_settings, "allowed_storage_roots", None),
        [Path(base_settings.storage_path).expanduser().resolve()],
    )
    source_roots = _parse_allowed_roots(
        getattr(base_settings, "allowed_source_roots", None),
        [Path.cwd().resolve(), CACHE_DIR.resolve()],
    )

    def _update_task(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with lock:
            existing = task_store.get(task_id, {})
            existing.update(payload)
            task_store[task_id] = existing
            _persist_task_store(task_store_path, task_store)
            return dict(existing)

    def _settings_with_overrides(req: IndexRequest) -> ArchitextSettings:
        overrides = {}
        if req.llm_provider:
            overrides["llm_provider"] = req.llm_provider
        if req.embedding_provider:
            overrides["embedding_provider"] = req.embedding_provider
        if overrides:
            return base_settings.model_copy(update=overrides)
        return base_settings

    def _resolve_storage_path(raw_path: Optional[str]) -> str:
        candidate = Path(raw_path or base_settings.storage_path).expanduser().resolve()
        if not _is_within_any(candidate, storage_roots):
            raise HTTPException(status_code=400, detail="storage must be within allowed roots")
        return str(candidate)

    def _resolve_output_dir(raw_path: Optional[str]) -> Optional[str]:
        if not raw_path:
            return None
        candidate = Path(raw_path).expanduser().resolve()
        if not _is_within_any(candidate, source_roots):
            raise HTTPException(status_code=400, detail="output_dir must be within allowed roots")
        return str(candidate)

    def _validate_source_dir(path: Path) -> None:
        if not path.exists():
            raise HTTPException(status_code=400, detail="source path does not exist")
        if not path.is_dir():
            raise HTTPException(status_code=400, detail="source path must be a directory")
        if not _is_within_any(path, source_roots):
            raise HTTPException(status_code=400, detail="source path must be within allowed roots")

    def _run_index(task_id: str, req: IndexRequest, storage_path: str):
        def progress_update(info: Dict[str, Any]):
            _update_task(task_id, {"progress": info})
        
        _update_task(task_id, {"status": "running", "storage_path": storage_path})
        try:
            task_settings = _settings_with_overrides(req)
            initialize_settings(task_settings)
            progress_update({"stage": "resolving", "message": "Resolving source..."})
            source_path = resolve_source(req.source, use_cache=not req.no_cache, ssh_key=req.ssh_key)
            _validate_source_dir(source_path)

            file_paths = gather_index_files(str(source_path), progress_callback=progress_update)

            if req.dry_run:
                _update_task(
                    task_id,
                    {
                        "status": "completed",
                        "documents": len(file_paths),
                        "storage_path": storage_path,
                        "dry_run": True,
                    },
                )
                return

            create_index_from_paths(
                file_paths,
                storage_path,
                progress_callback=progress_update,
                settings=task_settings,
            )
            _update_task(
                task_id,
                {
                    "status": "completed",
                    "documents": len(file_paths),
                    "storage_path": storage_path,
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            _update_task(
                task_id,
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "storage_path": storage_path,
                },
            )

    def _submit_task(req: IndexRequest, storage_path: str) -> str:
        task_id = str(uuid4())
        _update_task(task_id, {"status": "queued", "storage_path": storage_path})

        future = executor.submit(_run_index, task_id, req, storage_path)

        def _finalize(fut):
            try:
                fut.result()
            except Exception:  # pylint: disable=broad-except
                pass

        future.add_done_callback(_finalize)
        _update_task(task_id, {"future": future})
        return task_id

    def _submit_analysis_task(task_name: str, payload: TaskRequest) -> str:
        task_id = str(uuid4())
        _update_task(task_id, {"status": "queued", "task": task_name})

        def _run():
            def progress_update(info: Dict[str, Any]):
                _update_task(task_id, {"progress": info})

            _update_task(task_id, {"status": "running"})
            storage_path = _resolve_storage_path(payload.storage)
            try:
                if task_name == "analyze-structure":
                    result = analyze_structure(
                        storage_path=storage_path if not payload.source else None,
                        source_path=_resolve_task_source(payload.source, source_roots),
                        depth=payload.depth or "shallow",
                        output_format=payload.output_format,
                        progress_callback=progress_update,
                    )
                elif task_name == "tech-stack":
                    result = tech_stack(
                        storage_path=storage_path if not payload.source else None,
                        source_path=_resolve_task_source(payload.source, source_roots),
                        output_format=payload.output_format,
                        progress_callback=progress_update,
                    )
                elif task_name == "detect-anti-patterns":
                    result = detect_anti_patterns(
                        storage_path=storage_path if not payload.source else None,
                        source_path=_resolve_task_source(payload.source, source_roots),
                        progress_callback=progress_update,
                    )
                elif task_name == "health-score":
                    result = health_score(
                        storage_path=storage_path if not payload.source else None,
                        source_path=_resolve_task_source(payload.source, source_roots),
                        progress_callback=progress_update,
                    )
                elif task_name == "impact-analysis":
                    if not payload.module:
                        raise ValueError("module is required for impact analysis")
                    result = impact_analysis(
                        module=payload.module,
                        storage_path=storage_path if not payload.source else None,
                        source_path=_resolve_task_source(payload.source, source_roots),
                        progress_callback=progress_update,
                    )
                elif task_name == "refactoring-recommendations":
                    result = refactoring_recommendations(
                        storage_path=storage_path if not payload.source else None,
                        source_path=_resolve_task_source(payload.source, source_roots),
                        progress_callback=progress_update,
                    )
                elif task_name == "generate-docs":
                    result = generate_docs(
                        storage_path=storage_path if not payload.source else None,
                        source_path=_resolve_task_source(payload.source, source_roots),
                        output_dir=_resolve_output_dir(payload.output_dir),
                        progress_callback=progress_update,
                    )
                elif task_name == "detect-vulnerabilities":
                    result = detect_vulnerabilities(
                        storage_path=storage_path if not payload.source else None,
                        source_path=_resolve_task_source(payload.source, source_roots),
                        progress_callback=progress_update,
                    )
                elif task_name == "logic-gap-analysis":
                    result = logic_gap_analysis(
                        storage_path=storage_path if not payload.source else None,
                        source_path=_resolve_task_source(payload.source, source_roots),
                        progress_callback=progress_update,
                    )
                elif task_name == "identify-silent-failures":
                    result = identify_silent_failures(
                        storage_path=storage_path if not payload.source else None,
                        source_path=_resolve_task_source(payload.source, source_roots),
                        progress_callback=progress_update,
                    )
                elif task_name == "security-heuristics":
                    result = security_heuristics(
                        storage_path=storage_path if not payload.source else None,
                        source_path=_resolve_task_source(payload.source, source_roots),
                        progress_callback=progress_update,
                    )
                elif task_name == "code-knowledge-graph":
                    result = code_knowledge_graph(
                        storage_path=storage_path if not payload.source else None,
                        source_path=_resolve_task_source(payload.source, source_roots),
                        progress_callback=progress_update,
                    )
                elif task_name == "synthesis-roadmap":
                    result = synthesis_roadmap(
                        storage_path=storage_path if not payload.source else None,
                        source_path=_resolve_task_source(payload.source, source_roots),
                        progress_callback=progress_update,
                    )
                else:
                    raise ValueError("Unknown task")

                _update_task(task_id, {"status": "completed", "result": result})
            except Exception as exc:  # pylint: disable=broad-except
                _update_task(
                    task_id,
                    {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()},
                )

        future = executor.submit(_run)
        future.add_done_callback(lambda fut: fut.result() if not fut.cancelled() else None)
        _update_task(task_id, {"future": future})
        return task_id

    app = FastAPI(title="Architext API", version="0.2.0")
    app.state.task_store = task_store
    app.state.settings = base_settings
    app.state.executor = executor
    app.state.task_store_path = str(task_store_path)

    if getattr(base_settings, "rate_limit_per_minute", 0) > 0:
        limiter = _RateLimiter(base_settings.rate_limit_per_minute)

        @app.middleware("http")
        async def rate_limit(request: Request, call_next):
            client_ip = request.client.host if request.client else "unknown"
            if not limiter.allow(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                )
            return await call_next(request)

    @app.on_event("shutdown")
    async def _shutdown_cleanup():
        with lock:
            for task_id, payload in task_store.items():
                future = payload.get("future")
                if future is not None and not future.done():
                    future.cancel()
                if payload.get("status") in {"queued", "running"}:
                    payload["status"] = "stale"
                    payload["note"] = "Server shutdown before completion"
            _persist_task_store(task_store_path, task_store)

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/index", status_code=202)
    async def start_index(request: IndexRequest) -> Dict[str, Any]:
        storage_path = _resolve_storage_path(request.storage)

        if request.background:
            task_id = _submit_task(request, storage_path)
            return {
                "task_id": task_id,
                "status": "queued",
                "storage_path": storage_path,
            }

        task_id = str(uuid4())
        _run_index(task_id, request, storage_path)
        return {"task_id": task_id, **task_store.get(task_id, {})}

    @app.get("/status/{task_id}")
    async def get_status(task_id: str) -> Dict[str, Any]:
        task = task_store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return {"task_id": task_id, **task}

    @app.get("/tasks")
    async def list_tasks() -> Dict[str, Any]:
        return {"tasks": [{"task_id": task_id, **data} for task_id, data in task_store.items()]}

    @app.post("/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str) -> Dict[str, Any]:
        task = task_store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        future = task.get("future")
        if future is None:
            raise HTTPException(status_code=400, detail="Task cannot be cancelled")

        cancelled = future.cancel()
        if cancelled:
            _update_task(task_id, {"status": "cancelled"})
        return {"task_id": task_id, "cancelled": cancelled}

    @app.post("/tasks/analyze-structure", status_code=202)
    async def analyze_structure_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("analyze-structure", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        storage_path = _resolve_storage_path(request.storage) if not request.source else None
        result = analyze_structure(
            storage_path=storage_path,
            source_path=_resolve_task_source(request.source, source_roots),
            depth=request.depth or "shallow",
            output_format=request.output_format,
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "analyze-structure"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/tech-stack", status_code=202)
    async def tech_stack_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("tech-stack", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        storage_path = _resolve_storage_path(request.storage) if not request.source else None
        result = tech_stack(
            storage_path=storage_path,
            source_path=_resolve_task_source(request.source, source_roots),
            output_format=request.output_format,
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "tech-stack"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/detect-anti-patterns", status_code=202)
    async def detect_anti_patterns_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("detect-anti-patterns", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = detect_anti_patterns(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "detect-anti-patterns"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/health-score", status_code=202)
    async def health_score_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("health-score", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = health_score(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "health-score"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/impact-analysis", status_code=202)
    async def impact_analysis_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("impact-analysis", request)
            return {"task_id": task_id, "status": "queued"}

        if not request.module:
            raise HTTPException(status_code=400, detail="module is required")

        task_id = str(uuid4())
        result = impact_analysis(
            module=request.module,
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "impact-analysis"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/refactoring-recommendations", status_code=202)
    async def refactoring_recommendations_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("refactoring-recommendations", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = refactoring_recommendations(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "refactoring-recommendations"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/generate-docs", status_code=202)
    async def generate_docs_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("generate-docs", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = generate_docs(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
            output_dir=_resolve_output_dir(request.output_dir),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "generate-docs"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/dependency-graph", status_code=202)
    async def dependency_graph_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("dependency-graph", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = dependency_graph_export(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
            output_format=request.output_format,
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "dependency-graph"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/test-coverage", status_code=202)
    async def test_coverage_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("test-coverage", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = test_coverage_analysis(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "test-coverage"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/detect-patterns", status_code=202)
    async def detect_patterns_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("detect-patterns", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = architecture_pattern_detection(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "detect-patterns"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/diff-architecture", status_code=202)
    async def diff_architecture_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("diff-architecture", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = diff_architecture_review(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "diff-architecture"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/onboarding-guide", status_code=202)
    async def onboarding_guide_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("onboarding-guide", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = onboarding_guide(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "onboarding-guide"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/detect-vulnerabilities", status_code=202)
    async def detect_vulnerabilities_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("detect-vulnerabilities", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = detect_vulnerabilities(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "detect-vulnerabilities"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/logic-gap-analysis", status_code=202)
    async def logic_gap_analysis_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("logic-gap-analysis", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = logic_gap_analysis(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "logic-gap-analysis"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/identify-silent-failures", status_code=202)
    async def identify_silent_failures_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("identify-silent-failures", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = identify_silent_failures(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "identify-silent-failures"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/security-heuristics", status_code=202)
    async def security_heuristics_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("security-heuristics", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = security_heuristics(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "security-heuristics"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/code-knowledge-graph", status_code=202)
    async def code_knowledge_graph_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("code-knowledge-graph", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = code_knowledge_graph(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "code-knowledge-graph"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/tasks/synthesis-roadmap", status_code=202)
    async def synthesis_roadmap_task(request: TaskRequest) -> Dict[str, Any]:
        if request.background:
            task_id = _submit_analysis_task("synthesis-roadmap", request)
            return {"task_id": task_id, "status": "queued"}

        task_id = str(uuid4())
        result = synthesis_roadmap(
            storage_path=_resolve_storage_path(request.storage) if not request.source else None,
            source_path=_resolve_task_source(request.source, source_roots),
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "synthesis-roadmap"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/query")
    async def run_query(request: QueryRequest) -> Dict[str, Any]:
        storage_path = _resolve_storage_path(request.storage)

        overrides = {}
        if request.enable_hybrid is not None:
            overrides["enable_hybrid"] = request.enable_hybrid
        if request.hybrid_alpha is not None:
            overrides["hybrid_alpha"] = request.hybrid_alpha
        if request.enable_rerank is not None:
            overrides["enable_rerank"] = request.enable_rerank
        if request.rerank_model:
            overrides["rerank_model"] = request.rerank_model
        if request.rerank_top_n is not None:
            overrides["rerank_top_n"] = request.rerank_top_n

        request_settings = base_settings.model_copy(update=overrides) if overrides else base_settings

        try:
            index = load_existing_index(storage_path)
            response = query_index(index, request.text, settings=request_settings)
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        reranked = bool(request_settings.enable_rerank)
        hybrid_enabled = bool(request_settings.enable_hybrid)

        if request.mode == "agent":
            payload = (
                to_agent_response_compact(response)
                if request.compact
                else to_agent_response(response)
            )
            payload["reranked"] = reranked
            payload["hybrid_enabled"] = hybrid_enabled
            return payload

        return {
            "answer": str(response),
            "sources": extract_sources(response),
            "mode": "human",
            "reranked": reranked,
            "hybrid_enabled": hybrid_enabled,
        }

    @app.post("/ask")
    async def run_ask(request: AskRequest) -> Dict[str, Any]:
        storage_path = _resolve_storage_path(request.storage)

        overrides = {}
        if request.enable_hybrid is not None:
            overrides["enable_hybrid"] = request.enable_hybrid
        if request.hybrid_alpha is not None:
            overrides["hybrid_alpha"] = request.hybrid_alpha
        if request.enable_rerank is not None:
            overrides["enable_rerank"] = request.enable_rerank
        if request.rerank_model:
            overrides["rerank_model"] = request.rerank_model
        if request.rerank_top_n is not None:
            overrides["rerank_top_n"] = request.rerank_top_n

        request_settings = base_settings.model_copy(update=overrides) if overrides else base_settings

        try:
            index = load_existing_index(storage_path)
            response = query_index(index, request.text, settings=request_settings)
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        payload = (
            to_agent_response_compact(response)
            if request.compact
            else to_agent_response(response)
        )
        payload["reranked"] = bool(request_settings.enable_rerank)
        payload["hybrid_enabled"] = bool(request_settings.enable_hybrid)
        return payload

    @app.get("/mcp/tools")
    async def mcp_tools() -> Dict[str, Any]:
        return {"tools": _mcp_tools_schema()}

    @app.post("/mcp/run")
    async def mcp_run(request: MCPRunRequest) -> Dict[str, Any]:
        tool = request.tool
        args = request.arguments or {}

        if tool == "architext.query":
            payload = QueryRequest(**args)
            return await run_query(payload)

        if tool == "architext.ask":
            payload = AskRequest(**args)
            return await run_ask(payload)

        if tool == "architext.task":
            task_name = args.get("task")
            payload = TaskRequest(
                storage=args.get("storage"),
                source=args.get("source"),
                output_format=args.get("output_format", "json"),
                depth=args.get("depth"),
                module=args.get("module"),
                output_dir=args.get("output_dir"),
                background=False,
            )

            if task_name == "analyze-structure":
                result = analyze_structure(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                    depth=payload.depth or "shallow",
                    output_format=payload.output_format,
                )
            elif task_name == "tech-stack":
                result = tech_stack(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                    output_format=payload.output_format,
                )
            elif task_name == "detect-anti-patterns":
                result = detect_anti_patterns(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                )
            elif task_name == "health-score":
                result = health_score(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                )
            elif task_name == "impact-analysis":
                result = impact_analysis(
                    module=payload.module or "",
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                )
            elif task_name == "refactoring-recommendations":
                result = refactoring_recommendations(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                )
            elif task_name == "generate-docs":
                result = generate_docs(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                    output_dir=_resolve_output_dir(payload.output_dir),
                )
            elif task_name == "dependency-graph":
                result = dependency_graph_export(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                    output_format=payload.output_format,
                )
            elif task_name == "test-coverage":
                result = test_coverage_analysis(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                )
            elif task_name == "detect-patterns":
                result = architecture_pattern_detection(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                )
            elif task_name == "diff-architecture":
                result = diff_architecture_review(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                )
            elif task_name == "onboarding-guide":
                result = onboarding_guide(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                )
            elif task_name == "detect-vulnerabilities":
                result = detect_vulnerabilities(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                )
            elif task_name == "logic-gap-analysis":
                result = logic_gap_analysis(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                )
            elif task_name == "identify-silent-failures":
                result = identify_silent_failures(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                )
            elif task_name == "security-heuristics":
                result = security_heuristics(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                )
            elif task_name == "code-knowledge-graph":
                result = code_knowledge_graph(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                )
            elif task_name == "synthesis-roadmap":
                result = synthesis_roadmap(
                    storage_path=_resolve_storage_path(payload.storage) if not payload.source else None,
                    source_path=_resolve_task_source(payload.source, source_roots),
                )
            else:
                raise HTTPException(status_code=400, detail="Unknown task")

            return {"task": task_name, "result": result}

        raise HTTPException(status_code=400, detail="Unknown MCP tool")

    @app.post("/query/diagnostics")
    async def query_diagnostics(request: QueryRequest) -> Dict[str, Any]:
        from src.indexer import _tokenize, _keyword_score
        
        storage_path = _resolve_storage_path(request.storage)

        overrides = {}
        if request.enable_hybrid is not None:
            overrides["enable_hybrid"] = request.enable_hybrid
        if request.hybrid_alpha is not None:
            overrides["hybrid_alpha"] = request.hybrid_alpha
        if request.enable_rerank is not None:
            overrides["enable_rerank"] = request.enable_rerank
        if request.rerank_model:
            overrides["rerank_model"] = request.rerank_model
        if request.rerank_top_n is not None:
            overrides["rerank_top_n"] = request.rerank_top_n

        request_settings = base_settings.model_copy(update=overrides) if overrides else base_settings
        
        try:
            index = load_existing_index(storage_path)
            retriever = index.as_retriever(similarity_top_k=10)
            from llama_index.core.schema import QueryBundle
            nodes = retriever.retrieve(QueryBundle(query_str=request.text))
            
            diagnostics = []
            for node in nodes:
                content = node.node.get_content()[:200]
                keyword_score = _keyword_score(request.text, node.node.get_content())
                vector_score = node.score or 0.0
                
                diagnostics.append({
                    "file": node.metadata.get("file_path", "unknown"),
                    "content_preview": content,
                    "vector_score": vector_score,
                    "keyword_score": keyword_score,
                    "hybrid_score": 0.7 * vector_score + 0.3 * keyword_score,
                    "query_tokens": _tokenize(request.text),
                    "matched_tokens": list(set(_tokenize(request.text)) & set(_tokenize(node.node.get_content())))
                })
            
            return {
                "query": request.text,
                "results": diagnostics,
                "hybrid_enabled": request_settings.enable_hybrid,
                "rerank_enabled": request_settings.enable_rerank,
            }
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
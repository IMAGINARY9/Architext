"""FastAPI server exposing Architext indexing and query capabilities."""
from __future__ import annotations

from contextlib import asynccontextmanager
import traceback
import json
from pathlib import Path
import threading
from typing import Any, Dict, Optional
from uuid import uuid4

from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import JSONResponse

from src.config import ArchitextSettings, load_settings
from src.indexer import (
    create_index_from_paths,
    gather_index_files,
    initialize_settings,
)
from src.ingestor import resolve_source, CACHE_DIR
from src.indexer_components.factories import resolve_collection_name
from src.api.indices import build_indices_router
from src.api.mcp import build_mcp_router
from src.api.querying import build_query_router
from src.api.tasks import build_tasks_router
from src.api.tasks_service import (
    AnalysisTaskService,
    load_task_store,
    persist_task_store,
    resolve_task_store_path,
)
from src.api.schemas import (
    HealthResponse,
    IndexPreviewResponse,
    IndexRequest,
    IndexStartResponse,
    MCPRunRequest,
    ProvidersResponse,
    QueryRequest,
    TaskCancelResponse,
    TaskListSummaryResponse,
    TaskRequest,
    TaskStatusResponse,
    TaskSummaryResponse,
)
from src.server_utils import (
    RateLimiter,
    build_mcp_tools_schema,
    clear_chroma_storage,
    now_iso,
    parse_allowed_roots,
    resolve_storage_path,
    validate_source_dir,
)
# Backwards-compatible re-exports for tests and external patching
from src.tasks import (  # noqa: F401
    analyze_structure,
    tech_stack,
    detect_anti_patterns,
    health_score,
    impact_analysis,
    dependency_graph_export,
    test_mapping_analysis,
    architecture_pattern_detection,
    detect_vulnerabilities,
    identify_silent_failures,
    security_heuristics,
    code_knowledge_graph,
    synthesis_roadmap,
)


# --- Route modules extracted to src/api/indices.py, src/api/querying.py, ---
# --- schemas to src/api/schemas.py, utilities to src/server_utils.py     ---


def create_app(settings: Optional[ArchitextSettings] = None) -> FastAPI:
    """Create a FastAPI app configured with Architext settings."""

    base_settings = settings or load_settings()
    initialize_settings(base_settings)

    task_store_path = resolve_task_store_path(getattr(base_settings, "task_store_path", None))
    task_store: Dict[str, Dict[str, Any]] = load_task_store(task_store_path)
    lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=2)

    storage_roots = parse_allowed_roots(
        getattr(base_settings, "allowed_storage_roots", None),
        [Path(base_settings.storage_path).expanduser().resolve()],
    )
    source_roots = parse_allowed_roots(
        getattr(base_settings, "allowed_source_roots", None),
        [Path.cwd().resolve(), CACHE_DIR.resolve()],
    )

    task_service = AnalysisTaskService(
        task_store=task_store,
        task_store_path=task_store_path,
        lock=lock,
        executor=executor,
        storage_roots=storage_roots,
        source_roots=source_roots,
        base_settings=base_settings,
    )

    def _settings_with_overrides(req: IndexRequest) -> ArchitextSettings:
        overrides: Dict[str, Any] = {}
        if req.llm_provider:
            overrides["llm_provider"] = req.llm_provider
        if req.embedding_provider:
            overrides["embedding_provider"] = req.embedding_provider
        if overrides:
            return base_settings.model_copy(update=overrides)
        return base_settings

    def _run_index(task_id: str, req: IndexRequest, storage_path: str):
        def progress_update(info: Dict[str, Any]):
            task_service.update_task(task_id, {"progress": info})

        task_service.update_task(task_id, {"status": "running", "storage_path": storage_path})
        try:
            task_settings = _settings_with_overrides(req)
            initialize_settings(task_settings)
            progress_update({"stage": "resolving", "message": "Resolving source..."})
            source_path = resolve_source(req.source, use_cache=not req.no_cache, ssh_key=req.ssh_key)
            validate_source_dir(source_path, source_roots)

            file_paths = gather_index_files(str(source_path), progress_callback=progress_update)
            # Ensure reindexing overwrites previous Chroma data for the same storage path.
            if task_settings.vector_store_provider == "chroma":
                clear_chroma_storage(storage_path)

            create_index_from_paths(
                file_paths,
                storage_path,
                progress_callback=progress_update,
                settings=task_settings,
            )
            task_service.update_task(
                task_id,
                {
                    "status": "completed",
                    "documents": len(file_paths),
                    "storage_path": storage_path,
                    "result": {
                        "documents": len(file_paths),
                        "storage_path": storage_path,
                        "updated_at": now_iso(),
                    },
                },
            )
            # Persist basic index metadata to help index listing and debugging
            try:
                meta_file = Path(storage_path) / "index_metadata.json"
                existing_meta = {}
                if meta_file.exists():
                    try:
                        existing_meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                metadata = {
                    "source": req.source,
                    "created_at": existing_meta.get("created_at", now_iso()),
                    "updated_at": now_iso(),
                    "documents": len(file_paths),
                    "storage_path": storage_path,
                    "provider": task_settings.vector_store_provider,
                    "collection": resolve_collection_name(task_settings),
                }
                meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            except Exception:
                # Do not fail the task if metadata write fails
                pass
        except Exception as exc:  # pylint: disable=broad-except
            task_service.update_task(
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
        task_service.update_task(
            task_id,
            {
                "status": "queued",
                "task": "index",
                "storage_path": storage_path,
                "created_at": now_iso(),
            },
        )

        future = executor.submit(_run_index, task_id, req, storage_path)

        def _finalize(fut):
            try:
                fut.result()
            except Exception:  # pylint: disable=broad-except
                pass

        future.add_done_callback(_finalize)
        task_service.update_task(task_id, {"future": future})
        return task_id

    def _submit_analysis_task(task_name: str, payload: TaskRequest) -> str:
        task_id = str(uuid4())
        task_service.update_task(
            task_id,
            {
                "status": "queued",
                "task": task_name,
                "created_at": now_iso(),
            },
        )
        return task_service.submit_analysis_task(task_name, payload, task_id)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            with lock:
                for task_id, payload in task_store.items():
                    future = payload.get("future")
                    if future is not None and not future.done():
                        future.cancel()
                    if payload.get("status") in {"queued", "running"}:
                        payload["status"] = "stale"
                        payload["note"] = "Server shutdown before completion"
                persist_task_store(task_store_path, task_store)

    app = FastAPI(
        title="Architext API",
        version="0.2.0",
        description=(
            "Index repositories and query them using vector/hybrid search. "
            "Use /index or /index/preview to prepare data, then /query to retrieve answers."
        ),
        lifespan=lifespan,
    )
    app.state.task_store = task_store
    app.state.settings = base_settings
    app.state.executor = executor
    app.state.task_store_path = str(task_store_path)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(status="healthy")

    @app.get("/providers", response_model=ProvidersResponse, tags=["system"])
    async def get_providers() -> ProvidersResponse:
        """Get available providers and current configuration."""
        settings = app.state.settings
        return ProvidersResponse(
            llm_providers=["openai", "local"],
            embedding_providers=["huggingface", "openai"],
            vector_store_providers=["chroma", "qdrant", "pinecone", "weaviate"],
            default_llm_provider=settings.llm_provider,
            default_embedding_provider=settings.embedding_provider,
            default_storage_path=settings.storage_path,
            allowed_source_roots=settings.allowed_source_roots.split(",") if settings.allowed_source_roots else None,
            allowed_storage_roots=settings.allowed_storage_roots.split(",") if settings.allowed_storage_roots else None,
        )

    if getattr(base_settings, "rate_limit_per_minute", 0) > 0:
        limiter = RateLimiter(base_settings.rate_limit_per_minute)

        @app.middleware("http")
        async def rate_limit(request: Request, call_next):
            client_ip = request.client.host if request.client else "unknown"
            if not limiter.allow(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again later."},
                )
            return await call_next(request)

    # --- Index listing & metadata routes ---
    app.include_router(build_indices_router(storage_roots, base_settings))

    @app.post(
        "/index",
        status_code=202,
        response_model=IndexStartResponse,
        response_model_exclude_none=True,
        tags=["indexing"],
    )
    async def start_index(request: IndexRequest) -> IndexStartResponse:
        storage_path = resolve_storage_path(
            request.storage, request.source, storage_roots, base_settings.storage_path
        )

        # Validate the request first before queuing/starting
        try:
            task_settings = _settings_with_overrides(request)
            initialize_settings(task_settings)
            source_path = resolve_source(request.source, use_cache=not request.no_cache, ssh_key=request.ssh_key)
            validate_source_dir(source_path, source_roots)
            file_paths = gather_index_files(str(source_path))

            # Check for potential issues that would prevent successful indexing
            if len(file_paths) == 0:
                raise ValueError("No indexable files found in source")
            if len(file_paths) > 50000:  # Reasonable upper limit
                raise ValueError(f"Too many files ({len(file_paths)}) - maximum supported is 50,000")

        except Exception as exc:
            # Return validation error immediately without creating a task
            raise HTTPException(status_code=400, detail=f"Validation failed: {str(exc)}") from exc

        if request.background:
            task_id = _submit_task(request, storage_path)
            return IndexStartResponse(
                task_id=task_id,
                status="queued",
                storage_path=storage_path,
                documents=len(file_paths),
                result=None,
                error=None,
                created_at=task_store.get(task_id, {}).get("created_at"),
            )

        task_id = str(uuid4())
        task_service.update_task(
            task_id,
            {
                "status": "queued",
                "task": "index",
                "storage_path": storage_path,
                "created_at": now_iso(),
            },
        )
        _run_index(task_id, request, storage_path)
        return IndexStartResponse(task_id=task_id, **task_store.get(task_id, {}))

    @app.post("/index/preview", response_model=IndexPreviewResponse, tags=["indexing"])
    async def preview_index(request: IndexRequest) -> IndexPreviewResponse:
        """Preview what would be indexed without actually creating the index."""
        try:
            resolved_path = resolve_source(request.source, use_cache=not request.no_cache)
            validate_source_dir(resolved_path, source_roots)

            # Gather files to analyze
            file_paths = gather_index_files(str(resolved_path))

            # Analyze file types
            file_types: Dict[str, int] = {}
            for file_path in file_paths:
                ext = Path(file_path).suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1

            # Check for potential issues
            warnings = []
            if len(file_paths) > 10000:
                warnings.append("Large number of files detected - indexing may be slow")
            if len(file_paths) == 0:
                warnings.append("No indexable files found")

            return IndexPreviewResponse(
                source=request.source,
                resolved_path=str(resolved_path),
                documents=len(file_paths),
                file_types=file_types,
                warnings=warnings,
                would_index=len(file_paths) > 0
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get(
        "/status/{task_id}",
        response_model=TaskStatusResponse,
        response_model_exclude_none=True,
        tags=["tasks"],
    )
    async def get_status(task_id: str) -> TaskStatusResponse:
        task = task_store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return TaskStatusResponse(task_id=task_id, **task)

    @app.get(
        "/tasks",
        response_model=TaskListSummaryResponse,
        response_model_exclude_none=True,
        tags=["tasks"],
    )
    async def list_tasks() -> TaskListSummaryResponse:
        return TaskListSummaryResponse(
            tasks=[
                TaskSummaryResponse(task_id=task_id, **data)
                for task_id, data in task_store.items()
            ]
        )

    @app.post("/tasks/{task_id}/cancel", response_model=TaskCancelResponse, tags=["tasks"])
    async def cancel_task(task_id: str) -> TaskCancelResponse:
        task = task_store.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        future = task.get("future")
        if future is None:
            raise HTTPException(status_code=400, detail="Task cannot be cancelled")

        cancelled = future.cancel()
        if cancelled:
            task_service.update_task(task_id, {"status": "cancelled"})
        return TaskCancelResponse(task_id=task_id, cancelled=cancelled)

    # --- Analysis task routes ---
    app.include_router(
        build_tasks_router(
            _submit_analysis_task,
            task_service.run_analysis_task,
            task_service.update_task,
            TaskRequest,
            lambda: str(uuid4()),
        )
    )

    # --- Query routes ---
    query_router = build_query_router(storage_roots, base_settings)
    app.include_router(query_router)

    # --- MCP routes (uses query + task runners from above) ---
    app.include_router(
        build_mcp_router(
            lambda: build_mcp_tools_schema(storage_roots),
            query_router.run_query_for_mcp,  # type: ignore[attr-defined]
            task_service.run_analysis_task,
            QueryRequest,
            TaskRequest,
            MCPRunRequest,
            storage_roots,
            base_settings,
        )
    )

    return app


# Create app instance for uvicorn reload support
# This allows running: uvicorn src.server:app --reload
app = create_app()


if __name__ == "__main__":
    """Run the FastAPI app directly with uvicorn when executed as a module.

    Examples:
        python -m src.server --host 127.0.0.1 --port 8000
        uvicorn src.server:app --reload --host 127.0.0.1 --port 8000
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run Architext server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="DEPRECATED: Use 'uvicorn src.server:app --reload' instead")
    args = parser.parse_args()

    app = create_app()

    if args.reload:
        print("ERROR: Use 'uvicorn src.server:app --reload' for development with auto-reload.")
        print("The --reload flag with 'python -m src.server' is not supported.")
        import sys
        sys.exit(1)

    uvicorn.run(app, host=args.host, port=args.port)

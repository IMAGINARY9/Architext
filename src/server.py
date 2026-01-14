"""FastAPI server exposing Architext indexing and query capabilities."""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional
from uuid import uuid4

from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import ArchitextSettings, load_settings
from src.indexer import (
    create_index,
    initialize_settings,
    load_documents,
    load_existing_index,
    query_index,
)
from src.ingestor import resolve_source
from src.cli_utils import extract_sources, to_agent_response
from src.tasks import (
    analyze_structure,
    tech_stack,
    detect_anti_patterns,
    health_score,
    impact_analysis,
    refactoring_recommendations,
    generate_docs,
)


class IndexRequest(BaseModel):
    """Request payload to start indexing a repository."""

    source: str
    storage: Optional[str] = None
    no_cache: bool = False
    llm_provider: Optional[str] = Field(
        default=None, pattern="^(openai|gemini|local|anthropic)$"
    )
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


class TaskRequest(BaseModel):
    """Request payload for analysis tasks."""

    storage: Optional[str] = None
    source: Optional[str] = None
    output_format: str = Field(default="json", pattern="^(json|markdown|mermaid)$")
    depth: Optional[str] = Field(default="shallow", pattern="^(shallow|detailed|exhaustive)$")
    module: Optional[str] = None
    output_dir: Optional[str] = None
    background: bool = True


def create_app(settings: Optional[ArchitextSettings] = None) -> FastAPI:
    """Create a FastAPI app configured with Architext settings."""

    base_settings = settings or load_settings()
    initialize_settings(base_settings)

    task_store: Dict[str, Dict[str, Any]] = {}
    lock = threading.Lock()
    executor = ThreadPoolExecutor(max_workers=2)

    def _update_task(task_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with lock:
            existing = task_store.get(task_id, {})
            existing.update(payload)
            task_store[task_id] = existing
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

    def _run_index(task_id: str, req: IndexRequest, storage_path: str):
        def progress_update(info: Dict[str, Any]):
            _update_task(task_id, {"progress": info})
        
        _update_task(task_id, {"status": "running", "storage_path": storage_path})
        try:
            task_settings = _settings_with_overrides(req)
            initialize_settings(task_settings)
            progress_update({"stage": "resolving", "message": "Resolving source..."})
            source_path = resolve_source(req.source, use_cache=not req.no_cache)
            
            documents = load_documents(str(source_path), progress_callback=progress_update)

            if req.dry_run:
                _update_task(
                    task_id,
                    {
                        "status": "completed",
                        "documents": len(documents),
                        "storage_path": storage_path,
                        "dry_run": True,
                    },
                )
                return

            create_index(documents, storage_path, progress_callback=progress_update)
            _update_task(
                task_id,
                {
                    "status": "completed",
                    "documents": len(documents),
                    "storage_path": storage_path,
                },
            )
        except Exception as exc:  # pylint: disable=broad-except
            _update_task(
                task_id,
                {
                    "status": "failed",
                    "error": str(exc),
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
            storage_path = payload.storage or base_settings.storage_path
            try:
                if task_name == "analyze-structure":
                    result = analyze_structure(
                        storage_path=storage_path if not payload.source else None,
                        source_path=payload.source,
                        depth=payload.depth or "shallow",
                        output_format=payload.output_format,
                        progress_callback=progress_update,
                    )
                elif task_name == "tech-stack":
                    result = tech_stack(
                        storage_path=storage_path if not payload.source else None,
                        source_path=payload.source,
                        output_format=payload.output_format,
                        progress_callback=progress_update,
                    )
                elif task_name == "detect-anti-patterns":
                    result = detect_anti_patterns(
                        storage_path=storage_path if not payload.source else None,
                        source_path=payload.source,
                        progress_callback=progress_update,
                    )
                elif task_name == "health-score":
                    result = health_score(
                        storage_path=storage_path if not payload.source else None,
                        source_path=payload.source,
                        progress_callback=progress_update,
                    )
                elif task_name == "impact-analysis":
                    if not payload.module:
                        raise ValueError("module is required for impact analysis")
                    result = impact_analysis(
                        module=payload.module,
                        storage_path=storage_path if not payload.source else None,
                        source_path=payload.source,
                        progress_callback=progress_update,
                    )
                elif task_name == "refactoring-recommendations":
                    result = refactoring_recommendations(
                        storage_path=storage_path if not payload.source else None,
                        source_path=payload.source,
                        progress_callback=progress_update,
                    )
                elif task_name == "generate-docs":
                    result = generate_docs(
                        storage_path=storage_path if not payload.source else None,
                        source_path=payload.source,
                        output_dir=payload.output_dir,
                        progress_callback=progress_update,
                    )
                else:
                    raise ValueError("Unknown task")

                _update_task(task_id, {"status": "completed", "result": result})
            except Exception as exc:  # pylint: disable=broad-except
                _update_task(task_id, {"status": "failed", "error": str(exc)})

        future = executor.submit(_run)
        future.add_done_callback(lambda fut: fut.result() if not fut.cancelled() else None)
        _update_task(task_id, {"future": future})
        return task_id

    app = FastAPI(title="Architext API", version="0.2.0")
    app.state.task_store = task_store
    app.state.settings = base_settings
    app.state.executor = executor

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/index", status_code=202)
    async def start_index(request: IndexRequest) -> Dict[str, Any]:
        storage_path = request.storage or base_settings.storage_path

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
        storage_path = (request.storage or base_settings.storage_path) if not request.source else None
        result = analyze_structure(
            storage_path=storage_path,
            source_path=request.source,
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
        storage_path = (request.storage or base_settings.storage_path) if not request.source else None
        result = tech_stack(
            storage_path=storage_path,
            source_path=request.source,
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
            storage_path=(request.storage or base_settings.storage_path) if not request.source else None,
            source_path=request.source,
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
            storage_path=(request.storage or base_settings.storage_path) if not request.source else None,
            source_path=request.source,
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
            storage_path=(request.storage or base_settings.storage_path) if not request.source else None,
            source_path=request.source,
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
            storage_path=(request.storage or base_settings.storage_path) if not request.source else None,
            source_path=request.source,
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
            storage_path=(request.storage or base_settings.storage_path) if not request.source else None,
            source_path=request.source,
            output_dir=request.output_dir,
        )
        _update_task(task_id, {"status": "completed", "result": result, "task": "generate-docs"})
        return {"task_id": task_id, "status": "completed", "result": result}

    @app.post("/query")
    async def run_query(request: QueryRequest) -> Dict[str, Any]:
        storage_path = request.storage or base_settings.storage_path

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

        if request.mode == "agent":
            return to_agent_response(response)

        return {
            "answer": str(response),
            "sources": extract_sources(response),
            "mode": "human",
        }

    @app.post("/query/diagnostics")
    async def query_diagnostics(request: QueryRequest) -> Dict[str, Any]:
        from src.indexer import _tokenize, _keyword_score
        
        storage_path = request.storage or base_settings.storage_path
        
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
                "hybrid_enabled": base_settings.enable_hybrid,
                "rerank_enabled": base_settings.enable_rerank
            }
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app
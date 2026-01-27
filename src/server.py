"""FastAPI server exposing Architext indexing and query capabilities."""
from __future__ import annotations

from contextlib import asynccontextmanager
import os
import threading
import time
import traceback
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Union, Literal
from uuid import uuid4

from concurrent.futures import ThreadPoolExecutor
import asyncio
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
from src.indexer_components.factories import resolve_collection_name
from src.api.mcp import build_mcp_router
from src.api.tasks import build_tasks_router
from src.api.tasks_service import (
    AnalysisTaskService,
    load_task_store,
    persist_task_store,
    resolve_task_store_path,
)
from src.api_utils import extract_sources, to_agent_response, to_agent_response_compact
# Backwards-compatible re-exports for tests and external patching
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
    """Request payload to start indexing a repository.

    Note: fields provided here override the corresponding values loaded from the
    server configuration (env/config file) for this single request only.
    """

    source: str = Field(..., description="Local directory path or Git URL to index (e.g. './src' or 'https://github.com/user/repo.git'). Required.")
    storage: Optional[str] = Field(None, description="Directory to store the index; must be within allowed storage roots like './storage/my-index'. If omitted, server's configured storage_path is used.", examples=["./storage/my-custom-index"])
    no_cache: bool = Field(False, description="If true, do not use cached clone; always fetch remote repo anew. Only affects remote Git URLs.")
    ssh_key: Optional[str] = Field(None, description="Path to SSH key to use when cloning private repositories (e.g. '~/.ssh/id_rsa'). Only needed for private Git repos.", examples=["~/.ssh/id_rsa"])
    llm_provider: Optional[Literal["openai", "local"]] = Field(None, description="Override configured LLM provider for this request. Available: 'openai' (cloud) or 'local' (local LLM server).")
    embedding_provider: Optional[Literal["huggingface", "openai"]] = Field(None, description="Override configured embedding provider for this request. Available: 'huggingface' (local) or 'openai' (cloud).")
    background: bool = Field(True, description="Run indexing in background (queued task) if true; set false to run inline and wait for completion.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source": "./src",
                    "background": True,
                },
                {
                    "source": "https://github.com/user/repo.git",
                    "storage": "./storage/my-index",
                    "background": False,
                    "llm_provider": "openai",
                    "embedding_provider": "huggingface",
                },
            ]
        }
    }


class QueryRequest(BaseModel):
    """Request payload to query an existing index.

    Flags here override server configuration for the scope of this request.
    """

    text: str = Field(..., description="Query text to ask the index.")
    name: Optional[str] = Field(None, description="Name of the index to query (from /indices). If omitted, uses the default index.")
    mode: Literal["rag", "agent"] = Field(default="rag", description="Response mode: 'rag' (retrieval-augmented generation) for free text synthesized from retrieved sources, or 'agent' for structured output.")
    enable_hybrid: Optional[bool] = Field(None, description="Override to enable/disable hybrid (keyword+vector) search.")
    hybrid_alpha: Optional[float] = Field(None, description="Blend factor for hybrid scoring when enabled.")
    enable_rerank: Optional[bool] = Field(None, description="Override to enable/disable re-ranking of retrieved results.")
    rerank_model: Optional[str] = Field(None, description="Model to use for reranking when enabled.")
    rerank_top_n: Optional[int] = Field(None, description="How many top candidates to rerank.")
    compact: Optional[bool] = Field(None, description="When true in agent mode, returns a compact agent schema optimized for constrained contexts.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "How does auth work?",
                    "mode": "rag",
                    "name": "my-repo-index",
                },
                {
                    "text": "How does auth work?",
                    "mode": "agent",
                    "compact": True,
                    "name": "my-repo-index",
                },
            ]
        }
    }


class TaskRequest(BaseModel):
    """Request payload for analysis tasks."""

    storage: Optional[str] = Field(None, description="Optional index storage path to analyze.")
    source: Optional[str] = Field(None, description="Source repository path or URL to run the task against.")
    output_format: Literal["json", "markdown", "mermaid"] = Field(default="json", description="Output format for the task result.")
    depth: Optional[Literal["shallow", "detailed", "exhaustive"]] = Field(default="shallow", description="Analysis depth level.")
    module: Optional[str] = Field(None, description="Module name or path for module-specific tasks (e.g., impact analysis).")
    output_dir: Optional[str] = Field(None, description="Directory to write outputs for tasks that produce files.")
    background: bool = Field(True, description="Run the task in background when true; set false for inline execution.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "task": "tech-stack",
                    "storage": "./my-index",
                    "output_format": "json",
                    "depth": "shallow",
                },
                {
                    "task": "analyze-structure",
                    "source": "./src",
                    "output_format": "markdown",
                    "background": False,
                },
            ]
        }
    }


class MCPRunRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class IndexPreviewResponse(BaseModel):
    """Stable JSON schema for index preview responses."""
    source: str
    resolved_path: str
    documents: int
    file_types: Dict[str, int]
    warnings: List[str]
    would_index: bool
    error: Optional[str] = None


class ProvidersResponse(BaseModel):
    """Available providers and configuration options."""
    llm_providers: List[str] = Field(..., description="Available LLM providers")
    embedding_providers: List[str] = Field(..., description="Available embedding providers")
    vector_store_providers: List[str] = Field(..., description="Available vector store providers")
    default_llm_provider: str = Field(..., description="Currently configured default LLM provider")
    default_embedding_provider: str = Field(..., description="Currently configured default embedding provider")
    default_storage_path: str = Field(..., description="Default storage path for indices")
    allowed_source_roots: Optional[List[str]] = Field(None, description="Allowed source root directories (if configured)")
    allowed_storage_roots: Optional[List[str]] = Field(None, description="Allowed storage root directories (if configured)")


class QuerySource(BaseModel):
    """Schema for a single source in query responses."""
    file: str
    score: Optional[float] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None


class QueryResponse(BaseModel):
    """Stable JSON schema for RAG-mode query responses."""
    answer: str
    sources: List[QuerySource]
    mode: str = "rag"
    reranked: bool = False
    hybrid_enabled: bool = False


class AgentQueryResponse(BaseModel):
    """Stable JSON schema for agent-mode query responses."""
    answer: str
    confidence: Optional[float] = None
    sources: List[QuerySource]
    type: str
    reranked: bool = False
    hybrid_enabled: bool = False


class CompactAgentQueryResponse(BaseModel):
    """Compact schema for agent context windows."""
    answer: str
    confidence: Optional[float] = None
    sources: List[Dict[str, Any]]  # Simplified source format
    reranked: bool = False
    hybrid_enabled: bool = False


class IndexInfo(BaseModel):
    """Information about a single index."""
    name: str = Field(..., description="Name of the index (directory name)")
    path: str = Field(..., description="Full path to the index directory")
    documents: Optional[int] = Field(None, description="Number of documents in the index (None if not loaded)")
    provider: str = Field(..., description="Vector store provider (e.g., 'chroma', 'qdrant')")
    collection: str = Field(..., description="Collection name within the vector store")
    status: Optional[str] = Field(None, description="Status of the index ('available', 'load_error', etc.)")
    last_modified: Optional[str] = Field(None, description="ISO8601 timestamp of the last modification within the index directory")
    disk_usage_bytes: Optional[int] = Field(None, description="Approximate disk usage of the index directory in bytes")
    files_count: Optional[int] = Field(None, description="Number of files inside the index directory (useful to show directory is populated)")
    created_at: Optional[str] = Field(None, description="Creation timestamp if available in persisted metadata")
    updated_at: Optional[str] = Field(None, description="Last updated timestamp if available in persisted metadata")


class IndexListResponse(BaseModel):
    """Response for listing available indices."""
    indices: List[IndexInfo] = Field(default_factory=list, description="List of available indices", examples=[[
        {
            "name": "my-project",
            "path": "/path/to/storage/my-project",
            "documents": 42,
            "provider": "chroma",
            "collection": "architext_db"
        },
        {
            "name": "another-index",
            "path": "/path/to/storage/another-index", 
            "documents": 128,
            "provider": "chroma",
            "collection": "architext_db"
        }
    ]])


class IndexMetadataResponse(BaseModel):
    """Detailed metadata for a specific index."""
    name: str = Field(..., description="Name of the index")
    path: str = Field(..., description="Full path to the index directory")
    documents: Optional[int] = Field(None, description="Number of documents in the index")
    provider: Optional[str] = Field(None, description="Vector store provider")
    collection: Optional[str] = Field(None, description="Collection name within the vector store")
    status: str = Field(..., description="Status of the index")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata from the vector store")
    error: Optional[str] = Field(None, description="Error message if the index could not be loaded")
    last_modified: Optional[str] = Field(None, description="ISO8601 timestamp of the last modification within the index directory")
    disk_usage_bytes: Optional[int] = Field(None, description="Approximate disk usage of the index directory in bytes")
    files_count: Optional[int] = Field(None, description="Number of files inside the index directory (useful to show directory is populated)")
    created_at: Optional[str] = Field(None, description="Creation timestamp if available in persisted metadata")
    updated_at: Optional[str] = Field(None, description="Last updated timestamp if available in persisted metadata")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "my-project",
                    "path": "/path/to/storage/my-project",
                    "documents": 42,
                    "provider": "chroma",
                    "collection": "architext_db",
                    "status": "available"
                },
                {
                    "name": "broken-index",
                    "path": "/path/to/storage/broken-index",
                    "status": "error",
                    "error": "Failed to load index: database disk image is malformed"
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status of the service", examples=["healthy"])


class TaskStatusResponse(BaseModel):
    """Response for task status queries."""
    task_id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(..., description="Current status of the task")
    task: Optional[str] = Field(None, description="Name of the task being executed")
    storage_path: Optional[str] = Field(None, description="Storage path for indexing tasks")
    progress: Optional[Dict[str, Any]] = Field(None, description="Progress information for running tasks")
    result: Optional[Any] = Field(None, description="Result data for completed tasks")
    error: Optional[str] = Field(None, description="Error message for failed tasks")
    created_at: Optional[str] = Field(None, description="Task creation timestamp")

    model_config = {
        "exclude_none": True,
        "json_schema_extra": {
            "examples": [
                {
                    "task_id": "123e4567-e89b-12d3-a456-426614174000",
                    "status": "completed",
                    "task": "index",
                    "storage_path": "./my-index",
                    "result": {"documents": 42, "storage_path": "./my-index"},
                    "created_at": "2026-01-23T12:00:00Z",
                },
                {
                    "task_id": "123e4567-e89b-12d3-a456-426614174001",
                    "status": "running",
                    "task": "analyze-structure",
                    "progress": {"stage": "parsing", "completed": 15, "total": 42},
                    "created_at": "2026-01-23T12:01:00Z",
                },
            ]
        },
    }


class TaskListResponse(BaseModel):
    """Response for listing all tasks."""
    tasks: List[TaskStatusResponse] = Field(..., description="List of all tasks with their current status")


class TaskSummaryResponse(BaseModel):
    """Compact task summary for list views."""
    task_id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(..., description="Current status of the task")
    task: Optional[str] = Field(None, description="Name of the task being executed")
    storage_path: Optional[str] = Field(None, description="Storage path for indexing tasks")
    progress: Optional[Dict[str, Any]] = Field(None, description="Progress information for running tasks")
    error: Optional[str] = Field(None, description="Error message for failed tasks")
    created_at: Optional[str] = Field(None, description="Task creation timestamp")

    model_config = {
        "exclude_none": True,
    }


class TaskListSummaryResponse(BaseModel):
    """Response for listing all tasks (summary)."""
    tasks: List[TaskSummaryResponse] = Field(..., description="List of tasks with summary details")


class TaskCancelResponse(BaseModel):
    """Response for task cancellation."""
    task_id: str = Field(..., description="Unique identifier for the task")
    cancelled: bool = Field(..., description="Whether the task was successfully cancelled")


class IndexStartResponse(BaseModel):
    """Response for index creation requests."""
    task_id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(..., description="Current status of the task")
    storage_path: Optional[str] = Field(None, description="Storage path for indexing tasks")
    documents: Optional[int] = Field(None, description="Number of documents scheduled or indexed")
    result: Optional[Dict[str, Any]] = Field(None, description="Result data for completed tasks")
    error: Optional[str] = Field(None, description="Error message for failed tasks")
    created_at: Optional[str] = Field(None, description="Task creation timestamp")

    model_config = {
        "exclude_none": True,
        "json_schema_extra": {
            "examples": [
                {
                    "task_id": "123e4567-e89b-12d3-a456-426614174000",
                    "status": "queued",
                    "storage_path": "./my-index",
                    "documents": 42,
                    "created_at": "2026-01-23T12:00:00Z",
                }
            ]
        },
    }


class QueryDiagnosticsResult(BaseModel):
    """Single diagnostic entry for query diagnostics."""
    file: str
    content_preview: str
    vector_score: float
    keyword_score: float
    hybrid_score: float
    query_tokens: List[str]
    matched_tokens: List[str]


class QueryDiagnosticsResponse(BaseModel):
    """Response schema for query diagnostics."""
    query: str
    results: List[QueryDiagnosticsResult]
    hybrid_enabled: bool
    rerank_enabled: bool


class FileInfo(BaseModel):
    """Information about a file in an index."""
    file: str = Field(..., description="File path (normalized to forward slashes)")
    chunks: int = Field(..., description="Number of document chunks from this file")
    has_line_info: bool = Field(default=False, description="Whether line number information is available")


class IndexFilesResponse(BaseModel):
    """Response for listing files in an index."""
    index_name: str = Field(..., description="Name of the index")
    total_files: int = Field(..., description="Total number of unique files in the index")
    total_chunks: int = Field(..., description="Total number of document chunks")
    files: List[FileInfo] = Field(..., description="List of files with their chunk counts")


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


def _clear_chroma_storage(storage_path: str) -> None:
    """Clear Chroma sqlite files to avoid duplicate inserts on reindex."""
    base = Path(storage_path)
    if not base.exists():
        return
    for candidate in base.glob("chroma.sqlite3*"):
        if candidate.is_file():
            try:
                candidate.unlink()
            except Exception:
                continue


def _compute_source_hash(source: str) -> str:
    """Compute a stable hash for the source to use as storage subdirectory."""
    return hashlib.sha256(source.encode()).hexdigest()[:12]


def _compute_index_name(source: str) -> str:
    """Generate a meaningful index name from the source."""
    import re
    import hashlib
    from pathlib import Path
    
    if source.startswith(('http://', 'https://', 'git@', 'ssh://')):
        # For git URLs, extract repo name
        match = re.search(r'/([^/]+?)(\.git)?$', source)
        if match:
            name = match.group(1)
        else:
            name = 'repo'
    else:
        # For local paths, use basename
        name = Path(source).name
        if not name:
            name = 'local'
    
    # Sanitize: keep only alnum, _, -
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    # Limit length
    if len(name) > 20:
        name = name[:20]
    
    # Add short hash for uniqueness
    hash_part = hashlib.sha256(source.encode()).hexdigest()[:8]
    return f"{name}-{hash_part}"


def _dir_disk_usage(path: Path) -> int:
    """Return total size in bytes for files under path."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except Exception:
                continue
    return total


def _dir_last_modified(path: Path) -> Optional[str]:
    """Return ISO8601 timestamp of latest file modification in path, or None."""
    try:
        latest = max(p.stat().st_mtime for p in path.rglob("*"))
        return datetime.fromtimestamp(latest, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def _dir_file_count(path: Path) -> int:
    """Return number of files under path (recursive)."""
    count = 0
    for _ in path.rglob("*"):
        try:
            if _.is_file():
                count += 1
        except Exception:
            continue
    return count


def _mcp_tools_schema(storage_roots: List[Path]) -> List[Dict[str, Any]]:
    """Build MCP tools schema with available index names prefilled."""
    indices: List[str] = []
    try:
        for root in storage_roots:
            if not root.exists():
                continue
            # Allow the storage root itself to be an index
            if (root / "chroma.sqlite3").exists():
                indices.append(root.name)
            for item in root.iterdir():
                if item.is_dir():
                    chroma_path = item / "chroma.sqlite3"
                    if chroma_path.exists():
                        indices.append(item.name)
    except Exception:
        # If we can't list indices, continue without them.
        pass

    available = ", ".join(indices) if indices else "none found"
    return [
        {
            "name": "architext.query",
            "description": "Query an index and return agent/human output.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Query text to ask the index"},
                    "name": {"type": "string", "description": f"Name of the index to query. Available indices: {available}"},
                    "mode": {"type": "string", "enum": ["rag", "agent"], "description": "Response mode: 'rag' (retrieval-augmented generation) for free text, 'agent' for structured output"},
                    "compact": {"type": "boolean", "description": "When true in agent mode, returns a compact agent schema"},
                    "enable_hybrid": {"type": "boolean", "description": "Enable hybrid (keyword+vector) search"},
                    "hybrid_alpha": {"type": "number", "description": "Blend factor for hybrid scoring"},
                    "enable_rerank": {"type": "boolean", "description": "Enable re-ranking of retrieved results"},
                    "rerank_model": {"type": "string", "description": "Model to use for reranking"},
                    "rerank_top_n": {"type": "integer", "description": "How many top candidates to rerank"},
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
                    "task": {"type": "string", "description": "Task name (e.g., 'analyze-structure', 'tech-stack', 'detect-anti-patterns')"},
                    "storage": {"type": "string", "description": f"Optional index storage path to analyze. Available indices: {available}"},
                    "source": {"type": "string", "description": "Source repository path or URL to run the task against"},
                    "output_format": {"type": "string", "description": "Output format for the task result"},
                    "depth": {"type": "string", "description": "Analysis depth level"},
                    "module": {"type": "string", "description": "Module name for module-specific tasks"},
                    "output_dir": {"type": "string", "description": "Directory to write outputs for tasks that produce files"},
                },
                "required": ["task"],
            },
        },
        {
            "name": "architext.list_indices",
            "description": "List all available indices in storage.",
            "input_schema": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "architext.get_index_metadata",
            "description": "Get detailed metadata for a specific index.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "index_name": {"type": "string", "description": f"Name of the index. Available indices: {available}"},
                },
                "required": ["index_name"],
            },
        },
    ]



def create_app(settings: Optional[ArchitextSettings] = None) -> FastAPI:
    """Create a FastAPI app configured with Architext settings."""

    base_settings = settings or load_settings()
    initialize_settings(base_settings)

    task_store_path = resolve_task_store_path(getattr(base_settings, "task_store_path", None))
    task_store: Dict[str, Dict[str, Any]] = load_task_store(task_store_path)
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

    task_service = AnalysisTaskService(
        task_store=task_store,
        task_store_path=task_store_path,
        lock=lock,
        executor=executor,
        storage_roots=storage_roots,
        source_roots=source_roots,
        base_settings=base_settings,
    )

    def _now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _settings_with_overrides(req: IndexRequest) -> ArchitextSettings:
        overrides: Dict[str, Any] = {}
        if req.llm_provider:
            overrides["llm_provider"] = req.llm_provider
        if req.embedding_provider:
            overrides["embedding_provider"] = req.embedding_provider
        if overrides:
            return base_settings.model_copy(update=overrides)
        return base_settings

    def _query_settings_from_request(
        req: QueryRequest
    ) -> ArchitextSettings:
        overrides: Dict[str, Any] = {}
        if req.enable_hybrid is not None:
            overrides["enable_hybrid"] = req.enable_hybrid
        if req.hybrid_alpha is not None:
            overrides["hybrid_alpha"] = req.hybrid_alpha
        if req.enable_rerank is not None:
            overrides["enable_rerank"] = req.enable_rerank
        if req.rerank_model:
            overrides["rerank_model"] = req.rerank_model
        if req.rerank_top_n is not None:
            overrides["rerank_top_n"] = req.rerank_top_n
        return base_settings.model_copy(update=overrides) if overrides else base_settings

    def _ensure_sources_instruction(text: str) -> str:
        """Ensure queries explicitly request sources (file path and line ranges).

        If the user's question already asks for sources, return unchanged; otherwise
        append a short instruction that asks for file-level provenance.
        """
        lower = (text or "").lower()
        keywords = ["show sources", "show source", "file path", "file:", "quote line", "quote lines", "line", "lines", "sources", "cite"]
        if any(k in lower for k in keywords):
            return text
        instruction = "\n\nPlease include sources (file path and line ranges) for any code referenced, e.g., 'file.py:10-20'."
        return text + instruction

    def _resolve_storage_path(raw_path: Optional[str], source: Optional[str] = None) -> str:
        if raw_path:
            candidate = Path(raw_path).expanduser().resolve()
            if not _is_within_any(candidate, storage_roots):
                allowed_paths = ", ".join(str(root) for root in storage_roots)
                if raw_path in ["string", "path", "example", "null"]:
                    raise HTTPException(status_code=400, detail=f"storage path cannot be '{raw_path}' (placeholder value). Use a real path within allowed roots like '{allowed_paths}/my-index', or omit the field to use defaults.")
                raise HTTPException(status_code=400, detail=f"storage path '{candidate}' must be within allowed roots: {allowed_paths}")
            return str(candidate)
        else:
            if source:
                # Compute deterministic storage path based on source
                index_name = _compute_index_name(source)
                default_root = Path(base_settings.storage_path).expanduser().resolve()
                storage_path = default_root / index_name
                if not _is_within_any(storage_path, storage_roots):
                    raise HTTPException(status_code=400, detail=f"Computed storage path '{storage_path}' is not within allowed roots")
                return str(storage_path)
            else:
                # Use default storage path for queries
                candidate = Path(base_settings.storage_path).expanduser().resolve()
                if not _is_within_any(candidate, storage_roots):
                    raise HTTPException(status_code=400, detail=f"Default storage path '{candidate}' is not within allowed roots")
                return str(candidate)

    def _find_index_path_by_name(name: str) -> str:
        """Find the storage path for an index by its name."""
        for root in storage_roots:
            if not root.exists():
                continue
            # Check if root itself is the index
            if root.name == name and (root / "chroma.sqlite3").exists():
                return str(root)
            # Check subdirectories
            for item in root.iterdir():
                if item.is_dir() and item.name == name and (item / "chroma.sqlite3").exists():
                    return str(item)
        raise HTTPException(status_code=404, detail=f"Index with name '{name}' not found")

    def _list_available_indices() -> List[Dict[str, str]]:
        """Return a list of available indices as dicts with name and path."""
        found: List[Dict[str, str]] = []
        for root in storage_roots:
            if not root.exists():
                continue
            if (root / "chroma.sqlite3").exists():
                found.append({"name": root.name, "path": str(root)})
            for item in root.iterdir():
                if item.is_dir() and (item / "chroma.sqlite3").exists():
                    found.append({"name": item.name, "path": str(item)})
        return found

    def _resolve_index_storage(name: Optional[str]) -> str:
        """Resolve the storage path for an index.

        Behavior:
        - If name provided, find the matching index by name (404 if not found).
        - If name omitted:
          * If exactly one index exists in all storage roots, return it.
          * If none exist, raise 404 with helpful guidance.
          * If multiple exist, raise 400 requiring the caller to specify `name` and list available indices.
        """
        if name:
            return _find_index_path_by_name(name)

        candidates = _list_available_indices()
        if len(candidates) == 1:
            return candidates[0]["path"]
        if len(candidates) == 0:
            raise HTTPException(status_code=404, detail="No indices available. Create one with the /index endpoint.")
        # Multiple - present a clear error listing a subset of available names
        names = ", ".join(c["name"] for c in candidates[:10])
        raise HTTPException(
            status_code=400,
            detail=(
                f"Multiple indices available ({len(candidates)}). Specify the 'name' field to select one. "
                f"Available: {names}"
            ),
        )

    def _validate_source_dir(path: Path) -> None:
        if not path.exists():
            raise HTTPException(status_code=400, detail="source path does not exist")
        if not path.is_dir():
            raise HTTPException(status_code=400, detail="source path must be a directory")
        if not _is_within_any(path, source_roots):
            raise HTTPException(status_code=400, detail="source path must be within allowed roots")

    def _run_index(task_id: str, req: IndexRequest, storage_path: str):
        def progress_update(info: Dict[str, Any]):
            task_service.update_task(task_id, {"progress": info})
        
        task_service.update_task(task_id, {"status": "running", "storage_path": storage_path})
        try:
            task_settings = _settings_with_overrides(req)
            initialize_settings(task_settings)
            progress_update({"stage": "resolving", "message": "Resolving source..."})
            source_path = resolve_source(req.source, use_cache=not req.no_cache, ssh_key=req.ssh_key)
            _validate_source_dir(source_path)

            file_paths = gather_index_files(str(source_path), progress_callback=progress_update)
            # Ensure reindexing overwrites previous Chroma data for the same storage path.
            if task_settings.vector_store_provider == "chroma":
                _clear_chroma_storage(storage_path)

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
                        "updated_at": _now_iso(),
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
                    "created_at": existing_meta.get("created_at", _now_iso()),
                    "updated_at": _now_iso(),
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
                "created_at": _now_iso(),
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
                "created_at": _now_iso(),
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

    @app.get("/indices", response_model=IndexListResponse, tags=["indices"])
    async def list_indices() -> IndexListResponse:
        """List available indices in configured storage paths."""
        indices = []
        for root in storage_roots:
            if not root.exists():
                continue
            try:
                # Allow the storage root itself to be an index
                root_chroma = root / "chroma.sqlite3"
                if root_chroma.exists():
                    # Prefer metadata file if present
                    try:
                        meta_file = root / "index_metadata.json"
                        if meta_file.exists():
                            meta = json.loads(meta_file.read_text(encoding="utf-8"))
                            indices.append(IndexInfo(
                                name=root.name,
                                path=str(root),
                                documents=meta.get("documents"),
                                provider=base_settings.vector_store_provider,
                                collection=resolve_collection_name(base_settings),
                                status="available",
                                created_at=meta.get("created_at"),
                                updated_at=meta.get("updated_at"),
                                last_modified=_dir_last_modified(root),
                                disk_usage_bytes=_dir_disk_usage(root),
                                files_count=_dir_file_count(root),
                            ))
                            # Skip DB inspection if metadata exists
                            continue
                    except Exception:
                        pass

                    try:
                        index = load_existing_index(str(root))
                        doc_count = None
                        try:
                            stats = index.vector_store.client.get_collection(
                                resolve_collection_name(base_settings)
                            )
                            doc_count = stats.count if hasattr(stats, "count") else None
                        except Exception:
                            doc_count = None
                        indices.append(IndexInfo(
                            name=root.name,
                            path=str(root),
                            documents=doc_count,
                            provider=base_settings.vector_store_provider,
                            collection=resolve_collection_name(base_settings),
                            status="available",
                            last_modified=_dir_last_modified(root),
                            disk_usage_bytes=_dir_disk_usage(root),
                        ))
                    except Exception:
                        indices.append(IndexInfo(
                            name=root.name,
                            path=str(root),
                            documents=None,
                            provider=base_settings.vector_store_provider,
                            collection=resolve_collection_name(base_settings),
                            status="load_error",
                        ))
                for item in root.iterdir():
                    if item.is_dir():
                        # Check if this looks like a ChromaDB index directory
                        chroma_path = item / "chroma.sqlite3"
                        if chroma_path.exists():
                            # Try to read persisted metadata first (fast and avoids opening DB)
                            try:
                                meta_file = item / "index_metadata.json"
                                if meta_file.exists():
                                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                                    indices.append(IndexInfo(
                                        name=item.name,
                                        path=str(item),
                                        documents=meta.get("documents"),
                                        provider=base_settings.vector_store_provider,
                                        collection=resolve_collection_name(base_settings),
                                        status="available",
                                        created_at=meta.get("created_at"),
                                        updated_at=meta.get("updated_at"),
                                        last_modified=_dir_last_modified(item),
                                        disk_usage_bytes=_dir_disk_usage(item),
                                        files_count=_dir_file_count(item),
                                    ))
                                    continue
                            except Exception:
                                pass

                        # Try to get basic stats from the DB if metadata not available
                        try:
                            index = load_existing_index(str(item))
                            doc_count = None
                            try:
                                stats = index.vector_store.client.get_collection(
                                    resolve_collection_name(base_settings)
                                )
                                doc_count = stats.count if hasattr(stats, "count") else None
                            except Exception:
                                doc_count = None
                            indices.append(IndexInfo(
                                name=item.name,
                                path=str(item),
                                documents=doc_count,
                                provider=base_settings.vector_store_provider,
                                collection=resolve_collection_name(base_settings),
                                status="available",
                                last_modified=_dir_last_modified(item),
                                disk_usage_bytes=_dir_disk_usage(item),
                                files_count=_dir_file_count(item),
                            ))
                        except Exception:
                            # If we can't load it, still list it as available
                            indices.append(IndexInfo(
                                name=item.name,
                                path=str(item),
                                documents=None,
                                provider=base_settings.vector_store_provider,
                                collection=resolve_collection_name(base_settings),
                                status="load_error",
                            ))
            except Exception:
                continue
        return IndexListResponse(indices=indices)

    @app.get("/indices/{index_name}", response_model=IndexMetadataResponse, tags=["indices"])
    async def get_index_metadata(index_name: str) -> IndexMetadataResponse:
        """Get metadata for a specific index."""
        for root in storage_roots:
            candidates = [root / index_name]
            if root.name == index_name:
                candidates.append(root)
            for index_path in candidates:
                if not (index_path.exists() and index_path.is_dir()):
                    continue
                try:
                    # Prefer persisted metadata if present
                    meta_file = index_path / "index_metadata.json"
                    if meta_file.exists():
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                        metadata_dict = {
                            "name": index_name,
                            "path": str(index_path),
                            "documents": meta.get("documents"),
                            "provider": base_settings.vector_store_provider,
                            "collection": resolve_collection_name(base_settings),
                            "status": "available",
                            "metadata": meta,
                            "created_at": meta.get("created_at"),
                            "updated_at": meta.get("updated_at"),
                            "last_modified": _dir_last_modified(index_path),
                            "disk_usage_bytes": _dir_disk_usage(index_path),
                            "files_count": _dir_file_count(index_path),
                        }
                        return IndexMetadataResponse(**metadata_dict)

                    index = load_existing_index(str(index_path))
                    stats = index.vector_store.client.get_collection(
                        resolve_collection_name(base_settings)
                    )
                    doc_count = stats.count if hasattr(stats, 'count') else 0
                    
                    # Try to get more detailed metadata
                    metadata_dict = {
                        "name": index_name,
                        "path": str(index_path),
                        "documents": doc_count,
                        "provider": base_settings.vector_store_provider,
                        "collection": resolve_collection_name(base_settings),
                        "status": "available",
                        "last_modified": _dir_last_modified(index_path),
                        "disk_usage_bytes": _dir_disk_usage(index_path),
                        "files_count": _dir_file_count(index_path),
                    }
                    
                    # Add any additional metadata if available
                    if hasattr(stats, 'metadata') and stats.metadata:
                        metadata_dict["metadata"] = stats.metadata
                    
                    return IndexMetadataResponse(**metadata_dict)
                except Exception as exc:
                    return IndexMetadataResponse(
                        name=index_name,
                        path=str(index_path),
                        status="error",
                        error=str(exc),
                    )
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")

    @app.get("/indices/{index_name}/files", response_model=IndexFilesResponse, tags=["indices"])
    async def list_index_files(index_name: str) -> IndexFilesResponse:
        """List all files indexed in a specific index.
        
        Returns file paths with document chunk counts and metadata availability.
        This endpoint is useful for understanding index contents without performing queries.
        """
        storage_path = _resolve_index_storage(index_name)
        
        try:
            # Load the index to access its documents
            index = await asyncio.to_thread(load_existing_index, storage_path)
            
            # Get vector store from index
            vector_store = index.vector_store
            
            # Query all documents from ChromaDB
            from llama_index.core.schema import QueryBundle
            
            # Direct ChromaDB access for better performance
            metadatas = []
            if hasattr(vector_store, '_collection'):
                # Direct ChromaDB access
                collection = vector_store._collection
                result = collection.get(include=["metadatas"])
                metadatas = result.get("metadatas", [])
            else:
                # Fallback: use retriever with broad query
                retriever = index.as_retriever(similarity_top_k=10000)
                nodes = retriever.retrieve(QueryBundle(query_str="*"))
                metadatas = [node.node.metadata for node in nodes if hasattr(node, 'node')]
            
            # Aggregate file information
            file_counts = {}
            file_has_lines = {}
            
            for metadata in metadatas:
                if not isinstance(metadata, dict):
                    continue
                    
                file_path = metadata.get("file_path", "")
                if not file_path:
                    continue
                
                # Normalize path to forward slashes
                file_path = file_path.replace("\\", "/")
                
                file_counts[file_path] = file_counts.get(file_path, 0) + 1
                
                # Check if this file has line information
                if "start_line" in metadata or "end_line" in metadata:
                    file_has_lines[file_path] = True
            
            # Build file info list
            files = [
                FileInfo(
                    file=path,
                    chunks=count,
                    has_line_info=file_has_lines.get(path, False)
                )
                for path, count in sorted(file_counts.items())
            ]
            
            return IndexFilesResponse(
                index_name=index_name,
                total_files=len(files),
                total_chunks=sum(file_counts.values()),
                files=files
            )
            
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to list files: {str(exc)}") from exc

    @app.post(
        "/index",
        status_code=202,
        response_model=IndexStartResponse,
        response_model_exclude_none=True,
        tags=["indexing"],
    )
    async def start_index(request: IndexRequest) -> IndexStartResponse:
        storage_path = _resolve_storage_path(request.storage, request.source)

        # Validate the request first before queuing/starting
        try:
            task_settings = _settings_with_overrides(request)
            initialize_settings(task_settings)
            source_path = resolve_source(request.source, use_cache=not request.no_cache, ssh_key=request.ssh_key)
            _validate_source_dir(source_path)
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
                created_at=task_store.get(task_id, {}).get("created_at"),
            )

        task_id = str(uuid4())
        task_service.update_task(
            task_id,
            {
                "status": "queued",
                "task": "index",
                "storage_path": storage_path,
                "created_at": _now_iso(),
            },
        )
        _run_index(task_id, request, storage_path)
        return IndexStartResponse(task_id=task_id, **task_store.get(task_id, {}))

    @app.post("/index/preview", response_model=IndexPreviewResponse, tags=["indexing"])
    async def preview_index(request: IndexRequest) -> IndexPreviewResponse:
        """Preview what would be indexed without actually creating the index."""
        try:
            resolved_path = resolve_source(request.source, use_cache=not request.no_cache)
            _validate_source_dir(resolved_path)
            
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

    app.include_router(
        build_tasks_router(
            _submit_analysis_task,
            task_service.run_analysis_task,
            task_service.update_task,
            TaskRequest,
            lambda: str(uuid4()),
        )
    )

    async def _run_query_impl(payload: QueryRequest, client_request: Any) -> Union[QueryResponse, AgentQueryResponse, CompactAgentQueryResponse]:
        """Internal implementation of query that accepts an optional Request to support cancellation."""
        storage_path = _resolve_index_storage(payload.name)
        request_settings = _query_settings_from_request(payload)

        try:
            # Run blocking load and query in threads to avoid blocking the event loop.
            load_task = asyncio.create_task(asyncio.to_thread(load_existing_index, storage_path))
            # Poll for client disconnect while waiting for index to load
            while not load_task.done():
                if client_request and await client_request.is_disconnected():
                    load_task.cancel()
                    raise HTTPException(status_code=499, detail="Client disconnected")
                await asyncio.sleep(0.05)
            # Await completion (may raise)
            index = await load_task

            query_text = _ensure_sources_instruction(payload.text)
            query_task = asyncio.create_task(asyncio.to_thread(lambda: query_index(index, query_text, settings=request_settings)))
            while not query_task.done():
                if client_request and await client_request.is_disconnected():
                    query_task.cancel()
                    raise HTTPException(status_code=499, detail="Client disconnected")
                await asyncio.sleep(0.05)
            response = await query_task

        except HTTPException:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        reranked = bool(request_settings.enable_rerank)
        hybrid_enabled = bool(request_settings.enable_hybrid)

        if payload.mode == "rag":
            sources = extract_sources(response)
            return QueryResponse(
                answer=str(response),
                sources=[QuerySource(**s) for s in sources],
                reranked=reranked,
                hybrid_enabled=hybrid_enabled,
            )

        # Agent mode response — return explicit Pydantic models so FastAPI selects the
        # correct response schema rather than coercing to the first Union option.
        if payload.compact:
            result_payload = to_agent_response_compact(response, payload.text)
            result_payload["reranked"] = reranked
            result_payload["hybrid_enabled"] = hybrid_enabled
            return CompactAgentQueryResponse(**result_payload)

        result_payload = to_agent_response(response, payload.text)
        # Ensure Agent response has the required 'type' field
        result_payload["type"] = result_payload.get("type", "agent")
        result_payload["reranked"] = reranked
        result_payload["hybrid_enabled"] = hybrid_enabled
        return AgentQueryResponse(**result_payload)

    @app.post("/query", tags=["querying"])
    async def run_query(request: QueryRequest, req: Request) -> Union[QueryResponse, AgentQueryResponse, CompactAgentQueryResponse]:
        return await _run_query_impl(request, req)

    async def run_query_for_mcp(request: QueryRequest) -> Union[QueryResponse, AgentQueryResponse, CompactAgentQueryResponse]:
        """MCP-invokable wrapper that does not have a Request available."""
        return await _run_query_impl(request, None)

    app.include_router(
        build_mcp_router(
            lambda: _mcp_tools_schema(storage_roots),
            run_query_for_mcp,
            task_service.run_analysis_task,
            QueryRequest,
            TaskRequest,
            MCPRunRequest,
            storage_roots,
            base_settings,
        )
    )

    async def _query_diagnostics_impl(payload: QueryRequest, client_request: Any) -> QueryDiagnosticsResponse:
        from src.indexer import _tokenize, _keyword_score
        
        storage_path = _resolve_index_storage(payload.name)
        request_settings = _query_settings_from_request(payload)
        try:
            load_task = asyncio.create_task(asyncio.to_thread(load_existing_index, storage_path))
            while not load_task.done():
                if client_request and await client_request.is_disconnected():
                    load_task.cancel()
                    raise HTTPException(status_code=499, detail="Client disconnected")
                await asyncio.sleep(0.05)
            index = await load_task

            query_task = asyncio.create_task(asyncio.to_thread(lambda: index.as_retriever(similarity_top_k=10)))
            while not query_task.done():
                if client_request and await client_request.is_disconnected():
                    query_task.cancel()
                    raise HTTPException(status_code=499, detail="Client disconnected")
                await asyncio.sleep(0.05)
            retriever = await query_task

            from llama_index.core.schema import QueryBundle
            query_text = _ensure_sources_instruction(payload.text)
            nodes = retriever.retrieve(QueryBundle(query_str=query_text))

            diagnostics = []
            for node in nodes:
                content = node.node.get_content()[:200]
                keyword_score = _keyword_score(payload.text, node.node.get_content())
                vector_score = node.score or 0.0

                diagnostics.append({
                    "file": node.metadata.get("file_path", "unknown"),
                    "content_preview": content,
                    "vector_score": vector_score,
                    "keyword_score": keyword_score,
                    "hybrid_score": (
                        (request_settings.hybrid_alpha or 0.5) * vector_score
                        + (1.0 - (request_settings.hybrid_alpha or 0.5)) * keyword_score
                    ),
                    "query_tokens": _tokenize(payload.text),
                    "matched_tokens": list(set(_tokenize(payload.text)) & set(_tokenize(node.node.get_content())))
                })

            return QueryDiagnosticsResponse(
                query=payload.text,
                results=[QueryDiagnosticsResult(**entry) for entry in diagnostics],
                hybrid_enabled=bool(request_settings.enable_hybrid),
                rerank_enabled=bool(request_settings.enable_rerank),
            )
        except HTTPException:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post(
        "/query/diagnostics",
        response_model=QueryDiagnosticsResponse,
        response_model_exclude_none=True,
        tags=["querying"],
    )
    async def query_diagnostics(request: QueryRequest, req: Request) -> QueryDiagnosticsResponse:
        return await _query_diagnostics_impl(request, req)

    async def query_diagnostics_for_mcp(request: QueryRequest) -> QueryDiagnosticsResponse:
        return await _query_diagnostics_impl(request, None)

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

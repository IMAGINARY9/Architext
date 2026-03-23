"""Pydantic request/response schemas for the Architext API.

All API models are defined here as the single source of truth.
Server routes and routers import schemas from this module.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import ConfigDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    """Request payload to start indexing a repository.

    Note: fields provided here override the corresponding values loaded from the
    server configuration (env/config file) for this single request only.
    """

    source: str = Field(
        ...,
        description=(
            "Local directory path or Git URL to index "
            "(e.g. './src' or 'https://github.com/user/repo.git'). Required."
        ),
    )
    storage: Optional[str] = Field(
        None,
        description=(
            "Directory to store the index; must be within allowed storage roots "
            "like './storage/my-index'. If omitted, server's configured "
            "storage_path is used."
        ),
        examples=["./storage/my-custom-index"],
    )
    no_cache: bool = Field(
        False,
        description=(
            "If true, do not use cached clone; always fetch remote repo anew. "
            "Only affects remote Git URLs."
        ),
    )
    ssh_key: Optional[str] = Field(
        None,
        description=(
            "Path to SSH key to use when cloning private repositories "
            "(e.g. '~/.ssh/id_rsa'). Only needed for private Git repos."
        ),
        examples=["~/.ssh/id_rsa"],
    )
    llm_provider: Optional[Literal["openai", "local"]] = Field(
        None,
        description=(
            "Override configured LLM provider for this request. "
            "Available: 'openai' (cloud) or 'local' (local LLM server)."
        ),
    )
    embedding_provider: Optional[Literal["huggingface", "openai"]] = Field(
        None,
        description=(
            "Override configured embedding provider for this request. "
            "Available: 'huggingface' (local) or 'openai' (cloud)."
        ),
    )
    background: bool = Field(
        True,
        description=(
            "Run indexing in background (queued task) if true; "
            "set false to run inline and wait for completion."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"source": "./src", "background": True},
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
    name: Optional[str] = Field(
        None,
        description=(
            "Name of the index to query (from /indices). "
            "If omitted, uses the default index."
        ),
    )
    mode: Literal["rag", "agent"] = Field(
        default="rag",
        description=(
            "Response mode: 'rag' (retrieval-augmented generation) for free text "
            "synthesized from retrieved sources, or 'agent' for structured output."
        ),
    )
    enable_hybrid: Optional[bool] = Field(
        None, description="Override to enable/disable hybrid (keyword+vector) search."
    )
    hybrid_alpha: Optional[float] = Field(
        None, description="Blend factor for hybrid scoring when enabled."
    )
    enable_rerank: Optional[bool] = Field(
        None, description="Override to enable/disable re-ranking of retrieved results."
    )
    rerank_model: Optional[str] = Field(
        None, description="Model to use for reranking when enabled."
    )
    rerank_top_n: Optional[int] = Field(
        None, description="How many top candidates to rerank."
    )
    compact: Optional[bool] = Field(
        None,
        description=(
            "When true in agent mode, returns a compact agent schema "
            "optimized for constrained contexts."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "How does auth work?", "mode": "rag", "name": "my-repo-index"},
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

    storage: Optional[str] = Field(
        None, description="Optional index storage path to analyze."
    )
    source: Optional[str] = Field(
        None,
        description="Source repository path or URL to run the task against.",
    )
    output_format: Literal["json", "markdown", "mermaid"] = Field(
        default="json", description="Output format for the task result."
    )
    depth: Optional[Literal["shallow", "detailed", "exhaustive"]] = Field(
        default="shallow", description="Analysis depth level."
    )
    analysis_mode: Literal["full", "constrained"] = Field(
        default="full",
        description="Analysis profile. Use 'constrained' for lower-resource execution.",
    )
    constrained_max_files: int = Field(
        default=400,
        ge=50,
        le=5000,
        description="Maximum files to process when analysis_mode is 'constrained'.",
    )
    module: Optional[str] = Field(
        None,
        description="Module name or path for module-specific tasks (e.g., impact analysis).",
    )
    output_dir: Optional[str] = Field(
        None,
        description="Directory to write outputs for tasks that produce files.",
    )
    background: bool = Field(
        True,
        description="Run the task in background when true; set false for inline execution.",
    )

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
                    "analysis_mode": "constrained",
                    "background": False,
                },
            ]
        }
    }


class MCPRunRequest(BaseModel):
    """Request payload for MCP tool invocations."""

    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

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
    embedding_providers: List[str] = Field(
        ..., description="Available embedding providers"
    )
    vector_store_providers: List[str] = Field(
        ..., description="Available vector store providers"
    )
    default_llm_provider: str = Field(
        ..., description="Currently configured default LLM provider"
    )
    default_embedding_provider: str = Field(
        ..., description="Currently configured default embedding provider"
    )
    default_storage_path: str = Field(
        ..., description="Default storage path for indices"
    )
    allowed_source_roots: Optional[List[str]] = Field(
        None, description="Allowed source root directories (if configured)"
    )
    allowed_storage_roots: Optional[List[str]] = Field(
        None, description="Allowed storage root directories (if configured)"
    )


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
    documents: Optional[int] = Field(
        None,
        description="Number of documents in the index (None if not loaded)",
    )
    provider: str = Field(
        ..., description="Vector store provider (e.g., 'chroma', 'qdrant')"
    )
    collection: str = Field(
        ..., description="Collection name within the vector store"
    )
    status: Optional[str] = Field(
        None,
        description="Status of the index ('available', 'load_error', etc.)",
    )
    last_modified: Optional[str] = Field(
        None,
        description="ISO8601 timestamp of the last modification within the index directory",
    )
    disk_usage_bytes: Optional[int] = Field(
        None,
        description="Approximate disk usage of the index directory in bytes",
    )
    files_count: Optional[int] = Field(
        None,
        description="Number of files inside the index directory",
    )
    created_at: Optional[str] = Field(
        None, description="Creation timestamp if available in persisted metadata"
    )
    updated_at: Optional[str] = Field(
        None, description="Last updated timestamp if available in persisted metadata"
    )


class IndexListResponse(BaseModel):
    """Response for listing available indices."""

    indices: List[IndexInfo] = Field(
        default_factory=list,
        description="List of available indices",
        examples=[
            [
                {
                    "name": "my-project",
                    "path": "/path/to/storage/my-project",
                    "documents": 42,
                    "provider": "chroma",
                    "collection": "architext_db",
                },
                {
                    "name": "another-index",
                    "path": "/path/to/storage/another-index",
                    "documents": 128,
                    "provider": "chroma",
                    "collection": "architext_db",
                },
            ]
        ],
    )


class IndexMetadataResponse(BaseModel):
    """Detailed metadata for a specific index."""

    name: str = Field(..., description="Name of the index")
    path: str = Field(..., description="Full path to the index directory")
    documents: Optional[int] = Field(
        None, description="Number of documents in the index"
    )
    provider: Optional[str] = Field(None, description="Vector store provider")
    collection: Optional[str] = Field(
        None, description="Collection name within the vector store"
    )
    status: str = Field(..., description="Status of the index")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata from the vector store"
    )
    error: Optional[str] = Field(
        None, description="Error message if the index could not be loaded"
    )
    last_modified: Optional[str] = Field(
        None,
        description="ISO8601 timestamp of the last modification within the index directory",
    )
    disk_usage_bytes: Optional[int] = Field(
        None,
        description="Approximate disk usage of the index directory in bytes",
    )
    files_count: Optional[int] = Field(
        None, description="Number of files inside the index directory",
    )
    created_at: Optional[str] = Field(
        None, description="Creation timestamp if available in persisted metadata"
    )
    updated_at: Optional[str] = Field(
        None, description="Last updated timestamp if available in persisted metadata"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "my-project",
                    "path": "/path/to/storage/my-project",
                    "documents": 42,
                    "provider": "chroma",
                    "collection": "architext_db",
                    "status": "available",
                },
                {
                    "name": "broken-index",
                    "path": "/path/to/storage/broken-index",
                    "status": "error",
                    "error": "Failed to load index: database disk image is malformed",
                },
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(
        ..., description="Health status of the service", examples=["healthy"]
    )


class TaskStatusResponse(BaseModel):
    """Response for task status queries."""

    task_id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(..., description="Current status of the task")
    task: Optional[str] = Field(
        None, description="Name of the task being executed"
    )
    storage_path: Optional[str] = Field(
        None, description="Storage path for indexing tasks"
    )
    progress: Optional[Dict[str, Any]] = Field(
        None, description="Progress information for running tasks"
    )
    result: Optional[Any] = Field(
        None, description="Result data for completed tasks"
    )
    error: Optional[str] = Field(
        None, description="Error message for failed tasks"
    )
    created_at: Optional[str] = Field(
        None, description="Task creation timestamp"
    )

    model_config: ConfigDict = ConfigDict(  # type: ignore[misc,typeddict-unknown-key]
        exclude_none=True,
        json_schema_extra={
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
                    "progress": {
                        "stage": "parsing",
                        "completed": 15,
                        "total": 42,
                    },
                    "created_at": "2026-01-23T12:01:00Z",
                },
            ]
        },
    )


class TaskListResponse(BaseModel):
    """Response for listing all tasks."""

    tasks: List[TaskStatusResponse] = Field(
        ..., description="List of all tasks with their current status"
    )


class TaskSummaryResponse(BaseModel):
    """Compact task summary for list views."""

    task_id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(..., description="Current status of the task")
    task: Optional[str] = Field(
        None, description="Name of the task being executed"
    )
    storage_path: Optional[str] = Field(
        None, description="Storage path for indexing tasks"
    )
    progress: Optional[Dict[str, Any]] = Field(
        None, description="Progress information for running tasks"
    )
    error: Optional[str] = Field(
        None, description="Error message for failed tasks"
    )
    created_at: Optional[str] = Field(
        None, description="Task creation timestamp"
    )

    model_config: ConfigDict = ConfigDict(exclude_none=True)  # type: ignore[misc,typeddict-unknown-key]


class TaskListSummaryResponse(BaseModel):
    """Response for listing all tasks (summary)."""

    tasks: List[TaskSummaryResponse] = Field(
        ..., description="List of tasks with summary details"
    )


class TaskCancelResponse(BaseModel):
    """Response for task cancellation."""

    task_id: str = Field(..., description="Unique identifier for the task")
    cancelled: bool = Field(
        ..., description="Whether the task was successfully cancelled"
    )


class IndexStartResponse(BaseModel):
    """Response for index creation requests."""

    task_id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(..., description="Current status of the task")
    storage_path: Optional[str] = Field(
        None, description="Storage path for indexing tasks"
    )
    documents: Optional[int] = Field(
        None,
        description="Number of documents scheduled or indexed",
    )
    result: Optional[Dict[str, Any]] = Field(
        None, description="Result data for completed tasks"
    )
    error: Optional[str] = Field(
        None, description="Error message for failed tasks"
    )
    created_at: Optional[str] = Field(
        None, description="Task creation timestamp"
    )

    model_config: ConfigDict = ConfigDict(  # type: ignore[misc,typeddict-unknown-key]
        exclude_none=True,
        json_schema_extra={
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
    )


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


class IndexFileInfo(BaseModel):
    """Information about a file in an index."""

    file: str = Field(
        ..., description="File path (normalized to forward slashes)"
    )
    chunks: int = Field(
        ..., description="Number of document chunks from this file"
    )
    has_line_info: bool = Field(
        default=False,
        description="Whether line number information is available",
    )


class IndexFilesResponse(BaseModel):
    """Response for listing files in an index."""

    index_name: str = Field(..., description="Name of the index")
    total_files: int = Field(
        ..., description="Total number of unique files in the index"
    )
    total_chunks: int = Field(
        ..., description="Total number of document chunks"
    )
    files: List[IndexFileInfo] = Field(
        ..., description="List of files with their chunk counts"
    )

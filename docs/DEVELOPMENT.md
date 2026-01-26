# Architext: Developer Documentation

## 1. Project Structure

```
src/
  config.py      # Configuration management (Pydantic + .env)
  ingestor.py    # Repo resolution, cloning, and caching
  indexer.py     # LlamaIndex pipeline, ChromaDB, Reranking
  task_registry.py  # Task registry (single source of truth)
  tasks/         # Analysis tasks (split by domain)
  indexer_components/  # LLM, embeddings, vector store factories
  api/           # FastAPI routers and services
  server.py      # FastAPI application & task wiring
  api_utils.py   # Shared utilities for API responses and source extraction

tests/           # Pytest suite (see individual test modules)
storage/         # Default location for local vector indices
```

## 2. Core API Reference

### Configuration (`src/config.py`)
Centralized settings management.
```python
from src.config import ArchitextSettings
settings = ArchitextSettings()
# Access: settings.llm_provider, settings.storage_path
```

### Ingestion (`src/ingestor.py`)
Handles local paths and remote Git URLs.
```python
from src.ingestor import resolve_source
# Returns Path to local folder (cloning if necessary)
path = resolve_source("https://github.com/user/repo")
```

### Indexing (`src/indexer.py`)
Manages the RAG pipeline.
```python
from src.indexer import create_index, load_documents
docs = load_documents(path)
index = create_index(docs, storage_path="./storage")
```

### Server / API
The project exposes a FastAPI server with centralized task handling.
*   `POST /index`: Async indexing task.
*   `POST /index/preview`: Preview indexing plan (stable JSON schema).
*   `POST /query`: Semantic search query.
*   `POST /ask`: Agent-optimized query (compact schemas).
*   `GET /tasks/{id}`: Check status of async tasks.
*   `GET /mcp/tools`: Discover available tools (MCP-style).

### Stable JSON Schemas

Architext provides stable JSON schemas for agent integration. All responses follow these Pydantic models:

## Advanced / Static Defaults via Config

Advanced or static defaults can be put into `architext.config.json` in the project root (or `~/.architext/config.json`). Architext will automatically load this file if present and merge its values on top of `.env`/environment variables.

Example file (see `docs/advanced-config.json.example`):

```json
{
  "cache_enabled": false,
  "ssh_key": "~/.ssh/id_rsa",
  "enable_rerank": true,
  "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "rerank_top_n": 20,
  "enable_hybrid": true,
  "hybrid_alpha": 0.6
}
```

Add fields supported by `ArchitextSettings` (see `src/config.py`) such as `cache_enabled`, `ssh_key`, rerank/hybrid defaults, and other operational knobs.

#### Index Preview Schema
```json
{
  "source": "https://github.com/user/repo",
  "resolved_path": "/cache/repo-hash",
  "documents": 42,
  "file_types": {".py": 30, ".md": 5, ".txt": 7},
  "warnings": ["Remote repository will be cloned/cached locally"],
  "would_index": true
}
```

#### Query Response Schemas

**RAG Mode** (`POST /query` with `mode=rag`):
```json
{
  "answer": "The authentication is handled in auth.py...",
  "sources": [
    {
      "file": "src/auth.py",
      "score": 0.85,
      "start_line": 10,
      "end_line": 25
    }
  ],
  "mode": "rag",
  "reranked": false,
  "hybrid_enabled": true
}
```

**Agent Mode** (`POST /query` with `mode=agent`):
```json
{
  "answer": "Authentication uses JWT tokens...",
  "confidence": 0.85,
  "sources": [
    {
      "file": "src/auth.py", 
      "score": 0.85,
      "start_line": 10,
      "end_line": 25
    }
  ],
  "type": "Response",
  "reranked": false,
  "hybrid_enabled": true
}
```

**Compact Agent Mode** (`POST /ask` or `POST /query` with `compact=true`):
```json
{
  "answer": "JWT authentication...",
  "confidence": 0.85,
  "sources": [
    {"file": "src/auth.py", "line": 10, "score": 0.85}
  ],
  "reranked": false,
  "hybrid_enabled": true
}
```

## 3. roadmap & Implementation Status

### Completed Phases ✅
*   **Phase 1:** Core RAG Engine, CLI, Configuration.
*   **Phase 2:** API Server, Structured Outputs, Hybrid Search, Reranking.
*   **Phase 2.5:** Analysis Tasks (Structure, Anti-patterns, Health Score).
*   **Phase 2.9:** Security & Stability Remediation.
*   **Phase 3:** Semantic Intelligence (Active Auditing, AST Parsing).

### Future Opportunities (Phase 4+)
*   **Full MCP Server:** Formal implementation of the Model Context Protocol.
*   **Cloud Native:** First-class support for remote vector DBs (Pinecone/Qdrant) without adapters.
*   **Real-Time Watchers:** Incremental indexing on file changes.

## 4. Testing

Run the full test suite:
```bash
pytest tests/ -v
```

**Scope:**
*   Unit tests for CLI utils and Config.
*   Integration tests for ingestion and indexing.
*   Operational tests for the API server.

## 5. Tooling

Run linting and type checks:
```bash
python -m ruff check .
python -m mypy src
```

Enable pre-commit hooks:
```bash
pre-commit install
pre-commit run --all-files
```

## 6. Developing New Features
1.  **Add a Task:** Create a function in `src/tasks/<module>.py`, export it via `src/tasks/__init__.py`, register it in `src/task_registry.py`, and wire CLI support in `src/cli.py` if needed.
2.  **Add a Provider:** Update `src/config.py` Enum and `src/indexer_components/factories.py`.
3.  **Update Deps:** `pip freeze > requirements.txt`.

---
*For high-level project status, see [PROJECT_STATUS.md](PROJECT_STATUS.md).*

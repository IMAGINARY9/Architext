# Architext: Developer Documentation

## 1. Project Structure

```
src/
  config.py      # Configuration management (Pydantic + .env)
  ingestor.py    # Repo resolution, cloning, and caching
  indexer.py     # LlamaIndex pipeline, ChromaDB, Reranking
  tasks.py       # Analysis tasks (Phase 2.5 suite)
  server.py      # FastAPI application & Async task manager
  cli.py         # CLI entry point and command registration
  cli_utils.py   # Logging, formatting, model helpers

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
The project exposes a FastAPI server.
*   `POST /index`: Async indexing task.
*   `POST /query`: Semantic search query.
*   `GET /tasks/{id}`: Check status of async tasks.
*   `GET /mcp/tools`: Discover available tools (MCP-style).

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

## 5. Developing New Features
1.  **Add a Task:** Create a function in `src/tasks.py`, register it in `src/server.py` and `src/cli.py`.
2.  **Add a Provider:** Update `src/config.py` Enum and `src/indexer.py` factory logic.
3.  **Update Deps:** `pip freeze > requirements.txt`.

---
*For high-level project status, see [PROJECT_STATUS.md](PROJECT_STATUS.md).*

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

### Operator Workflow (Start Here)
Use this flow for reliable day-to-day operation:
1. Run `POST /index/preview` to verify file counts and warnings before indexing.
2. Run `POST /index` and monitor progress through `GET /tasks/{id}`.
3. Run `analyze-structure` and follow the `start_here` recommendations in the task output.
4. Run deeper analysis tasks after structural orientation (`health-score`, `detect-vulnerabilities`, etc.).

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

#### Structure Analysis JSON Additions
`analyze-structure` now returns a `start_here` list for onboarding guidance:

```json
{
  "format": "json",
  "summary": {"total_files": 42},
  "tree": {"src": {"__files__": ["server.py"]}},
  "start_here": [
    {"path": "README.md", "reason": "Project overview and quickstart"},
    {"path": "src/server.py", "reason": "Primary FastAPI app entry point"}
  ]
}
```

#### Constrained Analysis Mode
For low-resource environments, analysis tasks can use constrained mode via `TaskRequest`:

```json
{
  "source": "./src",
  "analysis_mode": "constrained",
  "constrained_max_files": 400
}
```

Contract note: response schemas are preserved. In `analyze-structure`, constrained mode adds summary metadata (`analysis_mode`, `files_skipped`) while keeping standard keys (`format`, `summary`, `tree`, `start_here`).

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
*   **Phase 1:** Core RAG engine and configuration foundation.
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
python -m pytest -q
```

**Scope:**
*   Unit tests for configuration and task modules.
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
1.  **Add a Task:** Create a class in `src/tasks/analysis/<module>.py` (extending `BaseTask`), export it via `src/tasks/__init__.py`, and register task entry points in `src/task_registry.py`.
2.  **Add a Provider:** Update `src/config.py` Enum and `src/indexer_components/factories.py`.
3.  **Update Deps:** `pip freeze > requirements.txt`.

## 7. Task Inventory Source of Truth

- Active task names, categories, and dependencies are canonical in `src/task_registry.py`.
- Task implementation architecture is organized under:
  - `src/tasks/core/`
  - `src/tasks/analysis/`
  - `src/tasks/orchestration/`

## 8. Improvement Backlog (Consolidated)

This backlog consolidates useful findings from internal retrospective work and comparative market analysis.

### Strategic Positioning and Guardrails

1. Keep Architext server-first and API-first; do not drift into CLI-first product behavior.
2. Prioritize agent-native outputs and orchestration use cases (enterprise automation, governance, AI toolchains).
3. Preserve complementarity with lightweight onboarding tools by improving integration surfaces, not by duplicating their UX model.
4. Treat portability and fallback behavior as reliability concerns, not only feature requests.

### Baseline Metrics and Targets

Comparative analysis highlighted practical operating expectations.

- Baseline references:
  - small repo indexing: ~3s
  - medium repo indexing: ~20-45s
  - query latency: ~2-5s
  - memory pressure risk grows materially on larger repositories
- Target direction (next iterations):
  1. keep small-repo indexing at or below current baseline.
  2. reduce medium-repo indexing tail latency.
  3. reduce peak memory during indexing and reranking workflows.
  4. document reproducible benchmark scenarios and results in release artifacts.

### P0/P1 Reliability and Architecture

1. Continue migrating heuristic/regex-heavy checks to AST-first analysis where feasible.
2. Keep strict path/input validation and rate limiting controls as non-negotiable defaults.
3. Improve performance strategy for very large repositories (batching, selective indexing, and better cache invalidation).

### P1/P2 Product and Developer Experience

1. Improve onboarding-oriented outputs (clear entry-point and "start-here" guidance for large codebases).
2. Define and document practical performance expectations for small/medium repositories.
3. Expand provider abstraction support and operational guidance for mixed local/cloud LLM setups.

### P2 Strategic Enhancements

1. Explore incremental re-indexing (file watcher/event-driven) to reduce full re-index frequency.
2. Improve dependency reasoning in dynamic import/injection scenarios.
3. Evaluate tighter IDE workflow integration patterns while preserving server-first architecture.
4. Expand provider abstraction path (including broader compatibility strategy such as LiteLLM-style routing).
5. Define optional lightweight analysis path for constrained environments while preserving core semantic mode.

### Comparative Analysis Action Mapping

| Comparative Finding | Architext Action |
|---|---|
| RAG depth is a strength, but setup/resource cost is high | Improve performance profile and operational defaults; publish benchmark matrix |
| Lightweight competitors win on onboarding ergonomics | Improve "start-here" recommendations and operator docs |
| Offline/heuristic fallback is a competitive advantage elsewhere | Investigate constrained-mode pathway without degrading primary semantic quality |
| Clear differentiation matters (platform vs onboarding tool) | Keep API-first architecture and focus on agent-intelligence workflows |

## 9. Temporary Execution Prompt Templates

For a temporary, repeatable execution workflow, see:

- `docs/TEMP_IMPROVEMENT_EXECUTION_PLAN.md`

---
*For high-level project status, see [PROJECT_STATUS.md](PROJECT_STATUS.md).*

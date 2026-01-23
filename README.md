# Architext

**An Intelligent "Codebase Architect" Agent**

Architext is a production-ready RAG (Retrieval-Augmented Generation) tool designed to index repositories, understand their architecture, and answer high-level questions. It serves as a "Cortex" for software architecture, useful for both human developers and AI Orchestrators.

![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Tests](https://img.shields.io/badge/Tests-92%2F92%20Passing-success)
![Phase](https://img.shields.io/badge/Phase-3%20Complete-blue)
![Version](https://img.shields.io/badge/Version-0.5.0-informational)

**[→ Read the Full Project Status Report](docs/PROJECT_STATUS.md)**

## Key Features

*   **Universal Ingestion:** Works with local folders, GitHub, GitLab, Gitea, and SSH URLs. Includes smart caching and deduplication.
*   **Deep Analysis:** Goes beyond text search. Includes tasks for **Structure Analysis**, **Anti-Pattern Detection**, **Tech Stack Inventory**, and **Health Scoring**.
*   **Semantic Intelligence:** Uses AST parsing and active auditing logic to find vulnerabilities and logic gaps.
*   **Agent-Native:** Designed to be used by *other* AI agents. features JSON output modes, structured tasks, and "Ask" APIs.
*   **Flexible & Secure:** Supports Local LLMs (Oobabooga, Ollama) and Cloud providers (OpenAI). Configuration-driven via `.env`.

## Documentation

*   **[PROJECT_STATUS.md](docs/PROJECT_STATUS.md)**: Detailed report on what has been delivered and tested.
*   **[PROJECT_RETROSPECTIVE.md](docs/PROJECT_RETROSPECTIVE.md)**: Critical assessment, self-reflection, and "dogfooding" analysis.
*   **[DEVELOPMENT.md](docs/DEVELOPMENT.md)**: Developer guide, API reference, and architecture roadmap.
*   **[MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)**: Compatibility notes for the task and server refactor.
*   **[RELEASE_NOTES.md](docs/RELEASE_NOTES.md)**: Highlights for the latest release.

## Getting Started

### Installation
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Quick Start

Start the server and use the HTTP API for all operations. Swagger UI: `http://localhost:8000/docs`.

```bash
# Start the server
python -m src.server --host 127.0.0.1 --port 8000
```

#### Core API Endpoints

- **Index a repository** — Creates a vector DB from a local path or remote git URL.

```bash
# Preview plan (recommended first step)
curl -X POST http://localhost:8000/index/preview \
  -H "Content-Type: application/json" \
  -d '{"source": "./src"}'

# Index with auto storage
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"source": "./src"}'

# Index with custom storage
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"source": "./src", "storage": "./my-index"}'
```

- **Query an index** — Ask a question against an existing index.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "How is authentication handled?", "storage": "./my-index", "format": "json"}'
```

#### Analysis & task suite (advanced / experimental)

These commands run static/heuristic analysis and produce JSON/markdown outputs. They are powerful but some outputs/flags may evolve between releases.

- Structure analysis (tree/mermaid/json):
```bash
curl -X POST http://localhost:8000/tasks/structure \
  -H "Content-Type: application/json" \
  -d '{"source": "./src", "output_format": "mermaid"}'
```

- Anti-pattern detection:
```bash
curl -X POST http://localhost:8000/tasks/anti-patterns \
  -H "Content-Type: application/json" \
  -d '{"source": "./src"}'
```

- Full audit (exports multiple artefacts):
```bash
curl -X POST http://localhost:8000/tasks/audit \
  -H "Content-Type: application/json" \
  -d '{"source": ".", "output": "./architext-audit"}'
```

For a full list of tasks and detailed usage, see `docs/DEVELOPMENT.md`.

> Note: Environment variables like `LLM_PROVIDER`, `OPENAI_API_KEY`, or local LLM endpoints should be set before starting the server. See **Configuration** below for examples. 🔒

**Preview Plan Schema** (for index preview):
```json
{
  "source": "./src",
  "resolved_path": "/path/to/src",
  "documents": 36,
  "file_types": {".py": 30, ".md": 5},
  "warnings": ["Remote repository will be cloned/cached locally"]
}
```

---


## Configuration

Duplicate `.env.example` to `.env` (if provided) or create one:

```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
# Or use local:
# LLM_PROVIDER=local
# OPENAI_API_BASE=http://localhost:5000/v1
```

**Optional JSON Config:** Create `architext.config.json` in your project root or `~/.architext/config.json` for advanced settings. Architext auto-detects and loads this file without CLI flags. Example:

```json
{
  "llm_provider": "openai",
  "embedding_provider": "openai",
  "enable_rerank": true
}
```

## API Schemas (for Agents)

Architext provides **stable JSON schemas** for reliable agent integration. All API responses follow consistent Pydantic models.

### Index Preview
```bash
curl -X POST http://localhost:8000/index/preview \
  -H "Content-Type: application/json" \
  -d '{"source": "https://github.com/user/repo"}'
```

**Response:**
```json
{
  "source": "https://github.com/user/repo",
  "resolved_path": "/cache/repo-hash",
  "documents": 42,
  "file_types": {".py": 30, ".md": 5},
  "warnings": ["Remote repository will be cloned"],
  "would_index": true
}
```

### Query Responses

**Human Mode:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "How does auth work?", "mode": "human"}'
```

**Agent Mode (Structured):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "How does auth work?", "mode": "agent"}'
```

**Compact Agent Mode:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"text": "How does auth work?"}'
```

All schemas include `confidence`, `sources`, `reranked`, and `hybrid_enabled` fields for consistent agent consumption.

## Testing

Architext maintains a high standard of code quality with the full test suite green.

```bash
pytest tests/ -v
```
# Architext

**An Intelligent "Codebase Architect" Agent**

Architext is a production-ready RAG (Retrieval-Augmented Generation) tool designed to index repositories, understand their architecture, and answer high-level questions. It serves as a "Cortex" for software architecture, useful for both human developers and AI Orchestrators.

![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Tests](https://img.shields.io/badge/Tests-319%2F319%20Passing-success)
![Phase](https://img.shields.io/badge/Phase-3%20Complete-blue)
![Version](https://img.shields.io/badge/Version-1.0.0-informational)

**[→ Read the Full Project Status Report](docs/PROJECT_STATUS.md)**

## Key Features

*   **Universal Ingestion:** Works with local folders, GitHub, GitLab, Gitea, and SSH URLs. Includes smart caching and deduplication.
*   **Deep Analysis:** Goes beyond text search. Includes tasks for **Structure Analysis**, **Anti-Pattern Detection**, **Tech Stack Inventory**, and **Health Scoring**.
*   **Semantic Intelligence:** Uses AST parsing and active auditing logic to find vulnerabilities and logic gaps.
*   **Agent-Native:** Designed to be used by *other* AI agents. features JSON output modes, structured tasks, and "Ask" APIs.
*   **Flexible & Secure:** Supports Local LLMs (Oobabooga, Ollama) and Cloud providers (OpenAI). Configuration-driven via `.env`.

## Documentation

*   **[PROJECT_STATUS.md](docs/PROJECT_STATUS.md)**: Detailed report on what has been delivered and tested.
*   **[DEVELOPMENT.md](docs/DEVELOPMENT.md)**: Developer guide, API reference, and architecture roadmap.
*   **[RELEASE_NOTES.md](docs/RELEASE_NOTES.md)**: Highlights for the latest release.
*   **[docs/research/README.md](docs/research/README.md)**: UX simulation assets and release gate operations.

## Audit Findings Snapshot

Key findings from internal retrospective and comparative review:

*   **Resolved critical risks:** Path traversal hardening, streaming ingestion (OOM mitigation), and improved task error visibility.
*   **Current strengths:** Strong task-based architecture, agent-native API outputs, and high reliability (`319/319` tests passing).
*   **Known limitations:** Large monorepo indexing cost, static index refresh model (no live incremental updates), and partial regex limitations outside deeper AST-driven coverage.
*   **Priority improvements:** Expand provider abstraction coverage, continue AST-first migration for heuristic checks, and improve fast onboarding workflows with clearer "start-here" outputs.

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
# Start the server (production)
python -m src.server --host 127.0.0.1 --port 8000

# Start the server with auto-reload (development)
uvicorn src.server:app --reload --host 127.0.0.1 --port 8000
```

### First Value Loop (10-15 Minutes)

Use this sequence for the highest first-run success rate:

1. Start the server and open `/docs`.
2. Run `POST /index/preview` for your target path.
3. Run `POST /index` only after preview confirms expected scope.
4. Poll `GET /status/{task_id}` until the index task completes.
5. Run `POST /query` with the index `name` (from `GET /indices`).

Example status check:

```bash
curl http://localhost:8000/status/<task_id>
```

Practical note:
- `POST /index` returns a `task_id`; use that same value when polling `GET /status/{task_id}`.
- Stop polling when status reaches a terminal state (`completed` or `failed`).
- Anti-pattern to avoid: do not copy placeholder schema values (for example `"string"`) into live requests.

Decision hint:
- Use `POST /query` for standard responses and broader query controls.
- Use `POST /query` with `compact=true` when you need compact, agent-optimized payloads.

For full request/response schemas and operator workflow details, see **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)**.

### Continuous UX Release Monitoring

UX evaluation is complete and now runs as release monitoring:

1. Produce/update a structured findings report for the cycle (see `docs/research/testing-cycle-findings-template.json`).
2. Run `.\\.venv\\Scripts\\python.exe scripts/run_ux_release_gate.py --findings-file docs/research/testing-cycle-findings-YYYY-MM-DD.json` for each release candidate.
3. If unresolved High findings are temporarily acceptable for release, run with `--allow-unresolved-high` and track explicit owners.
4. If any KPI threshold fails, run a full rerun using `docs/research/simulation-runbook.md`.
4. Review decisions, findings summary, and KPI snapshots in `docs/research/release-gate-log.md`.

Current thresholds:
- completion rate >= 85%
- time to first successful query <= 15 minutes
- wrong-endpoint attempts <= 1 median
- integration correctness >= 3/4

### First-Run Troubleshooting

| Symptom | Likely cause | Fastest fix |
|---|---|---|
| Validation errors in Swagger | Placeholder schema values (for example `"string"`) were submitted | Use the "Examples" dropdown payloads before sending requests |
| Query returns weak or empty results | Query was sent before index finished, or against the wrong index `name` | Confirm task completion via `GET /status/{task_id}` and select the correct index from `GET /indices` |
| Index scope is unexpectedly large | Source path was too broad for first run | Start with `./src` and use `POST /index/preview` before full indexing |

#### Core API Endpoints

- **Get available providers** — Discover supported LLM, embedding, and vector store providers.

```bash
curl http://localhost:8000/providers
```

- **Index a repository** — Creates a vector DB from a local path or remote git URL.

**Note:** When using Swagger UI (`/docs`), use the "Examples" dropdown for proper request formats. Avoid using placeholder values like "string" from the schema - these will result in validation errors.

```bash
# Preview plan (recommended first step)
curl -X POST http://localhost:8000/index/preview \
  -H "Content-Type: application/json" \
  -d '{"source": "./src"}'

# Index with auto storage (uses server defaults)
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"source": "./src"}'

# Index with custom storage
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"source": "./src", "storage": "./my-index"}'

# Index private repository with SSH
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"source": "git@github.com:user/private-repo.git", "ssh_key": "~/.ssh/id_rsa"}'
```

If index response includes `task_id`, poll `GET /status/{task_id}` until `completed` or `failed` before querying.

- **Query an index** — Ask a question against an existing index.

Schema-intent contrast:
- Prefer `POST /query` with `name` when you want standard controls and response modes.
- Prefer `POST /query` with `compact=true` for compact payloads in agent orchestration.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "How is authentication handled?", "name": "my-index", "mode": "rag"}'
```

#### Analysis & task suite (advanced / experimental)

These commands run static/heuristic analysis and produce JSON/markdown outputs. They are powerful but some outputs/flags may evolve between releases.

- Structure analysis (tree/mermaid/json):
```bash
curl -X POST http://localhost:8000/tasks/analyze-structure \
  -H "Content-Type: application/json" \
  -d '{"source": "./src", "output_format": "mermaid"}'
```

- Anti-pattern detection:
```bash
curl -X POST http://localhost:8000/tasks/detect-anti-patterns \
  -H "Content-Type: application/json" \
  -d '{"source": "./src"}'
```

- Parallel quality sweep (recommended audit-like pass):
```bash
curl -X POST http://localhost:8000/tasks/run-category/quality \
  -H "Content-Type: application/json" \
  -d '{"source": "./src", "max_workers": 4}'
```

For a full list of tasks, detailed usage, and response schemas, see **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)**.

> Note: Environment variables like `LLM_PROVIDER`, `OPENAI_API_KEY`, or local LLM endpoints should be set before starting the server. See **Configuration** below for examples. 🔒

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

For full schema documentation (index preview, query responses, index discovery, request-level overrides), see **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)**.

## Testing

Architext maintains a high standard of code quality with the full test suite green.

```bash
python -m pytest -q
```
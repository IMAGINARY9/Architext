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

### Quick Commands

A compact reference to the most-used commands. Some subcommands and analysis tasks are considered **advanced / experimental** and their flags may change — run `python -m src.cli <command> --help` for the latest options.

#### Core (stable) commands

- **Index a repository** — Creates a vector DB from a local path or remote git URL. Use `--storage` to set where the index is saved. For remote URLs Architext will auto-clone and cache the repo (use `--no-cache` to disable caching).

```bash
# Local (recommended: run with --dry-run first to preview)
python -m src.cli index ./src --storage ./my-index

# Remote (private repos: add --ssh-key)
python -m src.cli index https://github.com/psf/requests --storage ./requests-index
```

Key flags: `--dry-run` (preview files without persisting), `--no-cache`, `--ssh-key`, `--llm-provider`, `--embedding-provider`.

- **Query an index** — Ask a question against an existing index. Use `--storage` to point to the index and `--format json` for machine-readable output.

```bash
python -m src.cli query "How is authentication handled?" --storage ./my-index --format json
```

Advanced query flags: `--enable-hybrid`, `--hybrid-alpha`, `--enable-rerank`, `--rerank-model`, `--rerank-top-n`.

- **Run as a server (API)** — Start the FastAPI server. Use `--host`, `--port`, and `--reload` (dev only). Swagger UI: `http://localhost:8000/docs`.

```bash
python -m src.cli serve --host 127.0.0.1 --port 8000
```

#### Analysis & task suite (advanced / experimental)

These commands run static/heuristic analysis and produce JSON/markdown outputs. They are powerful but some outputs/flags may evolve between releases.

- Structure analysis (tree/mermaid/json):
```bash
python -m src.cli analyze-structure --source ./src --output-format mermaid
```

- Anti-pattern detection:
```bash
python -m src.cli detect-anti-patterns --source ./src
```

- Full audit (exports multiple artefacts):
```bash
python -m src.cli audit --source . --output ./architext-audit
```

For a full list of tasks and detailed usage, see `docs/DEVELOPMENT.md` and run `python -m src.cli <task> --help` before using in CI.

> Note: Environment variables like `LLM_PROVIDER`, `OPENAI_API_KEY`, or local LLM endpoints should be set before running commands that require an LLM. See **Configuration** below for examples. 🔒

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

## Testing

Architext maintains a high standard of code quality with the full test suite green.

```bash
pytest tests/ -v
```
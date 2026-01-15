# Architext

**An Intelligent "Codebase Architect" Agent**

Architext is a production-ready RAG (Retrieval-Augmented Generation) tool designed to index repositories, understand their architecture, and answer high-level questions. It serves as a "Cortex" for software architecture, useful for both human developers and AI Orchestrators.

![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Tests](https://img.shields.io/badge/Tests-86%2F86%20Passing-success)
![Phase](https://img.shields.io/badge/Phase-3%20Complete-blue)

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

## Getting Started

### Installation
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Quick Commands

**1. Index a Repository**
```bash
# Local
python -m src.cli index ./src --storage ./my-index

# Remote (Auto-clones and caches)
python -m src.cli index https://github.com/psf/requests --storage ./requests-index
```

**2. Ask a Question**
```bash
python -m src.cli query "How is authentication handled?" --storage ./my-index
```

**3. Run an Analysis Task**
```bash
# Generate a tree structure of the codebase
python -m src.cli analyze-structure ./my-index

# Check for architectural anti-patterns
python -m src.cli detect-anti-patterns ./my-index
```

**4. Run as a Server (API)**
```bash
python -m src.cli serve
# Swgger UI available at http://localhost:8000/docs
```

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

Architext maintains a high standard of code quality with 100% test pass rate on core features.

```bash
pytest tests/ -v
```

---
*Architext is a tool by [Your Name/Organization].*

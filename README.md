# Architext

**Architext** is an intelligent, local "Codebase Architect" tool. It indexes your source code to allow for high-level architectural queries using Retrieval-Augmented Generation (RAG).

## Features
*   **Privacy First:** Runs entirely locally using Oobabooga/Text-Generation-WebUI and local embedding models.
*   **Architectural Insight:** Ask questions like "How is authentication handled?" or "Where are the API routes defined?".
*   **Repo-wide Context:** Indexes your entire codebase, not just single files.

## Getting Started

### Prerequisites
1.  **Python 3.10+**
2.  **Local LLM Server:** A configured Oobabooga instance running with `--api`.
    *   See `docs/USAGE.md` (if available) or [Oobabooga GitHub](https://github.com/oobabooga/text-generation-webui).

### Installation
1.  Clone the repository.
2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage (Upcoming MVP)
*   **Check Connection:**
    ```bash
    python scripts/test_connection.py
    ```
*   **Index a Repo (Planned):**
    ```bash
    python -m src.cli index ./path/to/your/repo
    ```
*   **Query (Planned):**
    ```bash
    python -m src.cli query "Explain the database schema"
    ```

## Development
See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the roadmap and architecture details.

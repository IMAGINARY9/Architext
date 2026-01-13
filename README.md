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
*   **Index a Local Repository:**
    ```bash
    python -m src.cli index ./path/to/your/repo
    ```
*   **Index a Remote Repository (GitHub/GitLab):**
    ```bash
    # Clones to ~/.architext/cache/<repo_hash>
    python -m src.cli index https://github.com/user/repo
    
    # Or with SSH
    python -m src.cli index git@github.com:user/repo.git
    ```
*   **Query the Index:**
    ```bash
    python -m src.cli query "Explain the database schema"
    ```
*   **Clean Up Cached Repos:**
    ```bash
    python -m src.cli cache-cleanup --max-age 30
    ```

## Configuration

Architext is driven by environment variables (loaded from `.env` by default) via Pydantic settings. Key knobs:

* `LLM_PROVIDER` (`local`|`openai`|`gemini`|`anthropic`), `LLM_MODEL`, `OPENAI_API_BASE`, `OPENAI_API_KEY`, `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`
* `EMBEDDING_PROVIDER` (`huggingface`|`openai`), `EMBEDDING_MODEL_NAME`, `EMBEDDING_CACHE_DIR`
* `STORAGE_PATH` (default `./storage`), `CHUNK_SIZE`, `TOP_K`

Example `.env`:

```bash
LLM_PROVIDER=local
LLM_MODEL=local-model
OPENAI_API_BASE=http://127.0.0.1:5000/v1
OPENAI_API_KEY=local
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
STORAGE_PATH=./storage
```

CLI can point to another env file:

```bash
python -m src.cli --env-file ./dev.env index ./repo
```

## Development
See [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the roadmap and architecture details.

# Architext: Implementation Plan & Roadmap

## Project Overview
**Architext** is a local RAG (Retrieval-Augmented Generation) tool designed to index codebase architectures and answer high-level questions (e.g., "How does the auth flow work?", "Where are the API endpoints defined?").

**Core Philosophy:** Start simple with loose text chunking, then evolve into a sophisticated "Code Knowledge Graph" using AST parsing.

## Tech Stack (MVP)
*   **Language:** Python 3.10+
*   **Orchestrator:** LlamaIndex
*   **LLM:** Local Oobabooga API (OpenAI-compatible)
*   **Embeddings:** `sentence-transformers/all-mpnet-base-v2` (Local HuggingFace)
*   **Vector Database:** ChromaDB (Persisted locally)
*   **Interface:** CLI (Phase 1) -> Streamlit (Phase 2)

---

## Phase 1: The MVP (Proof of Concept)
**Goal:** Prove we can index a local folder and ask questions about it using a local LLM.

### 1.1 Environment Setup
*   [x] Verify Oobabooga API is running at `http://127.0.0.1:5000/v1`.
*   [x] Install dependencies: `llama-index`, `chromadb`, `sentence-transformers`, `llama-index-llms-openai-like`.

### 1.2 Core Indexing Logic (`src/indexer.py`)
*   [x] Implement `load_documents(path)`: Recursively read files (.py, .md, .js, etc.).
*   [x] Implement `create_index(documents)`: 
    *   Use `HuggingFaceEmbedding` for vectorization.
    *   Store in `ChromaDB` (saved to `./storage`).
*   [x] Implement `query_index(query)`: Retrieve top-k chunks and generate response.

### 1.3 CLI Interface (`src/cli.py`)
*   [x] Command: `python -m src.cli index <path_to_repo>`
*   [x] Command: `python -m src.cli query "Where is the login logic?"`

### 1.4 Connection Verification
*   [x] Script: `scripts/test_connection.py` to ensure LlamaIndex can talk to Oobabooga.

### 1.5 Testing & Validation (Reliability & Security)
*   [x] **Infrastructure**: Installed `pytest` and `pytest-mock`.
*   [x] **Unit Tests**:
    *   Verify `load_documents` respects `.gitignore` and ignores `.git/`, `.env`, and private keys (Security).
    *   Test `initialize_settings` performs graceful fallback if model is missing.
*   [x] **Integration Tests**:
    *   [x] Mock LLM/Embedding calls to test `initialize_settings` and embedding setup without heavy model loading.
    *   [x] Verify full `indexer` -> `CLI` pipeline via mocked LLM and embeddings, and ensure CLI exit codes on errors.

**Recommended next testing steps (short-term):**
- Add integration tests that run `python -m src.cli index` and `python -m src.cli query` with LLM & embedding calls mocked; assert exit codes and output.
- Add a CI workflow (GitHub Actions) that runs `pytest` and fails on regressions.
- Add code coverage reporting and set a reasonable coverage gate (e.g., 70%).

---

## Phase 2: The Polished Tool (Alpha)
**Goal:** Make it usable for daily development with a better UI and smarter context.

### 2.1 "Smart" Indexing
*   [ ] Switch from simple chunking to **AST-based chunking** (using `CodeSplitter`).
*   [ ] Preserve context: Ensure every chunk knows its filename and class name.

### 2.2 User Interface
*   [ ] Build a Streamlit app (`src/app.py`):
    *   Sidebar: Select repository to index.
    *   Main: Chat interface with "Source" dropdowns to show which files were used.

### 2.3 Incremental Updates
*   [ ] Track file modification times. Only re-index changed files to save time.

---

## Phase 3: Production / Advanced
**Goal:** Turn it into a powerful architectural analysis tool.

### 3.1 Graph capabilities
*   [ ] Integrate `llama-index` Graph integrations to map function calls.

### 3.2 IDE Integration
*   [ ] Create a VS Code extension that queries the local Architext server.

## Developer Setup
1.  **Clone Repo**: `git clone ...`
2.  **Env**: `python -m venv .venv` & `source .venv/bin/activate`
3.  **Install**: `pip install -r requirements.txt`
4.  **Run**: Ensure Oobabooga is running with `--api`.

# Architext: Implementation Plan & Roadmap (2.0)

## Project Overview
**Architext** is a universal architectural search engine designed to index any codebase (local or remote) and answer high-level questions. It acts as a specialized "Cortex" for software architecture, useful for both human developers and AI Orchestrators.

**Core Philosophy:** 
1.  **Universality:** Works with any language, any repo structure, anywhere (Local/GitHub/GitLab).
2.  **Headless & Composable:** Designed primarily as a CLI tool and an API/MCP Server for other agents, not a visual dashboard.
3.  **Model Agnostic:** Bring your own LLM (OpenAI, Gemini, Claude, Local Oobabooga/Ollama).

## Tech Stack (Revised)
*   **Language:** Python 3.10+
*   **Core Logic:** LlamaIndex / LangChain (orchestration)
*   **LLM Interface:** `LiteLLM` (Unified interface for OpenAI, Vertex, Bedrock, & Local)
*   **Ingestion:** `GitPython` (Remote cloning), Native File I/O
*   **Vector Database:** ChromaDB (Local/Server mode), extensible to Pinecone/Qdrant
*   **Interface:** CLI (`click`/`typer`) & API (`FastAPI`)
*   **Config:** `pydantic-settings` (.env management)

---

## Phase 1: The Pivot (Foundation) ✅
**Goal:** Transition from the rigid local MVP to a flexible, configuration-driven tool.

### 1.1 Configuration System ✅ **COMPLETED**
*   [x] Replace hardcoded Oobabooga/Path settings with `src/config.py` using Pydantic.
*   [x] Support `.env` loading for sensitivity (e.g., `OPENAI_API_KEY`, `GITHUB_TOKEN`).
*   [x] Allow switching "Providers" via config:
    *   `LLM_PROVIDER`: `openai` | `gemini` | `local` | `anthropic`
    *   `EMBEDDING_PROVIDER`: `huggingface` (local) | `openai`

### 1.2 Universal Ingestion (`src/ingestor.py`) ✅ **COMPLETED**
*   [x] Integrate `GitPython` for remote cloning.
*   [x] Detect input type: local path vs. GitHub/GitLab URL.
*   [x] Clone remote repos to cache directory (`~/.architext/cache/<repo_hash>`).
*   [x] Handle authentication for private repos via SSH Agent or PAT (Personal Access Token).
*   [x] Add `--repo` flag to CLI for remote indexing (changed to positional `source` arg).
*   [x] Validate repo accessibility before indexing.
*   [x] Add cleanup task for old cached repos.

### 1.3 Flexible LLM Backend & Tuning ✅ **COMPLETED**
*   [x] Abstract LLM calls using OpenAI-compatible interface (OpenAILike).
*   [x] **Inference Tuning**: Expose parameters (`temperature`, `max_tokens`) via config.
*   [x] **Prompt Customization**: Support user-defined System Prompts via the config file (crucial for optimizing across different models like DeepSeek vs. GPT-4).
*   [x] **RAG Optimization**: Make retrieval parameters (`chunk_size`, `top_k`) configurable.
*   [x] Verify connectivity with Local (Oobabooga) and OpenAI providers.
*   [ ] Add support for Gemini and Anthropic providers.

### 1.4 Core Refactor ✅ **COMPLETED**
*   [x] Decouple `indexer.py` from CLI print statements. Make it a library.
*   [x] Thread config through CLI with `--env-file` flag and storage path overrides.
*   [x] Add comprehensive test coverage for config-driven flows.
*   [x] Add ingestor module for universal repo handling (local + remote).
*   [ ] Implement "Lazy Loading" / Streaming for documents to handle larger repos without OOM errors.

### 1.5 End-to-End Integration Testing ✅ **COMPLETED**
*   [x] Create integration test: index local repo + query.
*   [x] Test with multi-language repositories (Python, TypeScript, Markdown).
*   [x] Validate embedding model loading and vector search.
*   [x] Test storage persistence and index reloading.
*   [x] Create multi-language test repos (Python, TypeScript, Java).
*   [x] Verify chunking preserves code semantics.

### 1.6 CLI Enhancements ✅ **COMPLETED**
*   [x] Add provider selection flags (`--llm-provider`, `--embedding-provider`).
*   [x] Add `--list-models` command to show available local models.
*   [x] Add `--dry-run` flag to preview indexing without persistence.
*   [x] Improve error messages and validation feedback.
*   [x] Add progress indicators for long-running operations.
*   [x] Add `--verbose` flag for debugging logs.
*   [x] Support custom output formats (plain text, JSON).

---

## Phase 2: API & Agent Integration ✅ (2.1-2.4 COMPLETE)
**Goal:** Expose Architext as a service that other agents can query.

### 2.1 The "Headless" Server (`src/server.py`) ✅ **COMPLETED**
*   [x] Implement a `FastAPI` service.
*   [x] Endpoints:
    *   `POST /index`: Trigger indexing of a repo URL or Trigger re-index.
    *   `POST /query`: Semantic search against a specific index.
    *   `GET /status`: Indexing progress.
    *   `GET /tasks`: List all tasks.
    *   `POST /tasks/{task_id}/cancel`: Cancel background tasks.
    *   `POST /query/diagnostics`: Hybrid scoring diagnostics.
*   [x] Thread pool-based async task execution.

### 2.2 Structured Outputs ✅ **COMPLETED**
*   [x] Implement **Dual-Mode Response**:
    *   **Human Mode**: Natural language summary with citations.
    *   **Agent Mode (JSON)**: Strict schema for machine parsing.
    *   *Schema Example:* `{ "answer": "...", "confidence": 0.9, "sources": [{"file": "auth.ts", "lines": [10, 20]}] }`
*   [x] Source extraction helper (`extract_sources`).
*   [x] Agent-optimized response formatter (`to_agent_response`).

### 2.3 CLI Enhancements ✅ **COMPLETED**
*   [x] Add `architext serve` to start the API.
*   [x] Add `--format json` flag to `query` command.
*   [x] Add hybrid search flags (`--enable-hybrid`, `--hybrid-alpha`).
*   [x] Add reranking flags (`--enable-rerank`, `--rerank-model`, `--rerank-top-n`).

### 2.4 Quality & Retrieval Optimization ✅ **COMPLETED**
*   [x] **Re-ranking (Cross-Encoders)**: Implement a second-stage retrieval step to re-rank the top-k results before passing them to the LLM. This significantly reduces noise and increases accuracy.
*   [x] **Hybrid Search**: Combine keyword search with semantic search for better finding of specific class/function names.
*   [x] **Progress Tracking**: Real-time indexing progress callbacks.
*   [x] **Cross-Encoder Caching**: Model caching to avoid reloading.
*   [x] **Configurable retrieval**: All hybrid/rerank parameters exposed via CLI and API.

### 2.5 Default Task Suite (Agent-Optimized Workflows)
**Goal:** Provide pre-built analysis tasks leveraging the indexed codebase, making Architext a reasoning engine for architecture, not just search.

**Core Default Tasks:**
*   [x] **Repository Structure Analysis**: Generate visual/JSON map of module organization, layer separation, and dependency flow.
    *   CLI: `architext analyze-structure <index_path>`
    *   API: `POST /tasks/analyze-structure`
    *   Output: Module tree, layer diagram (Mermaid), coupling metrics
*   [x] **Architectural Anti-Patterns Detection**: Identify circular dependencies, god objects, SoC violations, tight coupling.
    *   CLI: `architext detect-anti-patterns <index_path>`
    *   API: `POST /tasks/detect-anti-patterns`
    *   Output: List of issues with severity + suggested fixes
*   [x] **Technology Stack Inventory**: What frameworks/libraries are used where, with counts and distribution.
    *   CLI: `architext tech-stack <index_path>`
    *   API: `POST /tasks/tech-stack`
    *   Output: Structured list of frameworks, versions, usage breakdown
*   [x] **Architectural Health Scoring**: Rate modules by modularity, coupling, documentation coverage, testing gaps.
    *   CLI: `architext health-score <index_path>`
    *   API: `POST /tasks/health-score`
    *   Output: Numeric scores (0-100) with breakdown by category + improvement suggestions
*   [x] **Impact Analysis**: "If I change module X, which components are affected?"
    *   CLI: `architext impact-analysis <index_path> --module <module_name>`
    *   API: `POST /tasks/impact-analysis`
    *   Output: Dependency tree, list of affected components with confidence scores
*   [x] **Refactoring Recommendations**: Suggest architectural improvements with effort estimates and migration paths.
    *   CLI: `architext refactoring-recommendations <index_path>`
    *   API: `POST /tasks/refactoring-recommendations`
    *   Output: Prioritized list of refactoring opportunities with effort/benefit analysis
*   [x] **Documentation Compilation**: Generate architecture decision records (ADRs), module summaries, and system diagrams from code.
    *   CLI: `architext generate-docs <index_path> --output ./docs`
    *   API: `POST /tasks/generate-docs`
    *   Output: Markdown files, Mermaid diagrams, architecture summary

**Implementation Approach:**
*   Tasks are **parameterized** via request body: `{ "depth": "shallow|detailed|exhaustive", "output_format": "json|markdown|mermaid", ... }`
*   All tasks **reuse the existing RAG pipeline** — no new infrastructure needed.
*   Tasks can be **chained** for agent workflows: `index → analyze-structure → refactoring-recommendations → generate-docs`
*   Support **async task execution** for long-running analyses (with `/tasks/<task_id>/status` polling endpoint).

**Similar Features (Phase 2.5 Extensions):**
*   [x] **Dependency Graph Export**: Output in Mermaid, PlantUML, GraphQL formats for visualization tools.
*   [x] **Test Coverage Analysis**: Correlate test files to modules, identify gaps and testing priorities.
*   [x] **Architecture Pattern Detection**: Recognize and classify architectural patterns (MVC, microservices, monolith, plugin architecture, event-driven, etc.).
*   [x] **Diff-Based Architecture Review**: Compare two commits/branches: "What architectural changes happened? Are they aligned with the architecture guidelines?"
*   [x] **Onboarding Guide Generation**: Auto-generate "where to start reading the codebase" and navigation path based on stated purpose/role.

---

## Phase 3: Scale & Intelligence
**Goal:** Handle massive enterprise monorepos and enhance retrieval accuracy.

### 3.1 Advanced Storage Strategy
*   [ ] **Remote Vector Store Support**: Add adapters for:
    *   Pinecone / Qdrant (Cloud scaling).
    *   Remote Chroma (Team sharing).
*   [ ] **Persistent Indexes**: Allow naming/saving indexes (e.g., `architext index --name "backend-v1" ./src`).

### 3.2 Advanced Retrieval & High-Fidelity Indexing
*   [ ] **AST-Based Chunking**: Integrate `CodeSplitter` to respect function/class boundaries (Python, JS, Java) instead of arbitrary line breaks.
*   [ ] **Context Injection**: Decorate every node with metadata (filename, class path) to preserve context in isolated chunks.
*   [ ] **Code Knowledge Graph**: Integrate Graph stores to map function calls and dependencies, enabling complex "impact analysis" queries.

### 3.3 Agent Ecosystem
*   [ ] **MCP (Model Context Protocol) Server**: Wrap the API as an endpoint compatible with Claude Desktop / generic MCP clients.

---

## Performance & Optimization Guidelines
*   **Prompt Tuning**: Different models (e.g., DeepSeek Coder vs. GPT-4) require unique instruction styles. Always test the system prompt in `config.py` when switching models.
*   **Retrieval Knobs**: For large repositories, increase `chunk_size` to 1024 and use a Re-ranker to maintain precision.
*   **Local Quantization**: When running local LLMs, prefer **GGUF (Q4_K_M or Q5_K_M)** formats via Ollama or Oobabooga to balance inference speed with architectural reasoning capabilities.
*   **Embedding Models**: Use `mpnet-base-v2` for local precision or `text-embedding-3-small` for cloud-based cost efficiency.

## Quality Assurance & Best Practices
*   [ ] **Integration Tests**: Mock `LiteLLM` responses to test pipeline without real API costs.
*   [ ] **Security Scans**: Ensure `.git` history and `.env` files are strictly ignored during indexing.
*   [ ] **CI/CD**: GitHub Actions to run tests and linters (Ruff/Black).

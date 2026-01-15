# Architext: Implementation Plan & Roadmap (v2.1)

> **Status Update (Jan 2026):** Phases 1 and 2 are effectively **COMPLETE**. The project is now pivoting to **Remediation** (Phase 2.9) and **Semantic Intelligence** (Phase 3).

## Project Overview
**Architext** is a universal architectural search engine designed to index any codebase (local or remote) and answer high-level questions. It acts as a specialized "Cortex" for software architecture, useful for both human developers and AI Orchestrators.

**Core Philosophy:** 
1.  **Universality:** Works with any language, any repo structure, anywhere (Local/GitHub/GitLab).
2.  **Headless & Composable:** Designed primarily as a CLI tool and an API/MCP Server for other agents, not a visual dashboard.
3.  **Model Agnostic:** Bring your own LLM (OpenAI, Gemini, Claude, Local Oobabooga/Ollama).

## Tech Stack (Revised)
*   **Language:** Python 3.10+
*   **Core Logic:** LlamaIndex (RAG orchestration)
*   **LLM Interface:** OpenAI + OpenAI-compatible providers (via `llama-index-llms-openai` and `llama-index-llms-openai-like`)
*   **Ingestion:** `GitPython` (Remote cloning), Native File I/O
*   **Vector Database:** ChromaDB (Local/Server mode), with optional adapter scaffolding for Pinecone/Qdrant/Weaviate
*   **Parsing/Chunking:** `tree-sitter` + `tree-sitter-languages` (logical chunking by function/class when enabled)
*   **Interface:** CLI (`argparse`) & API (`FastAPI`)
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
*   [ ] Add support for Gemini and Anthropic providers (deferred; removed from config until LiteLLM integration).

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

---✅ Phase 2.9: Critical Remediation (Priority: Immediate)
**Goal:** Fix security vulnerabilities and stability risks identified during self-audit.

### 2.9.1 Security Hardening
*   [x] **Path Traversal Protection**: Validate `storage` and `source` inputs in `server.py` to prevent arbitrary file reads (e.g., `../../../etc/passwd`).
*   [x] **Rate Limiting**: Implement token bucket or request limiting on FastAPI endpoints to prevent DOS.
*   [x] **Input Sanitization**: Ensure `read_bytes` and file operations do not accept tainted paths from API requests.

### 2.9.2 Stability & Error Handling
*   [x] **Streaming Ingestion**: Replace full-memory `load_documents()` with a streaming/batch generator to prevent OOM on large (>1GB) repos.
*   [x] **Reranking Transparency**:
    *   Fail loudly if `--enable-rerank` is requested but model fails to load.
    *   Include `reranked: boolean` in response metadata so users know if fallback occurred.
*   [x] **Exception Visibility**: Capture and return stack traces in API error responses (instead of generic "failed").

### 2.9.3 Configuration Integrity
*   [x] **Fix Provider Mismatch**: `config.py` allows `gemini`/`anthropic`, but `indexer.py` crashes.
    *   *Action:* Implement `LiteLLM` integration OR remove unsupported options from Enum.
*   [x] **Task Durability**: Persist task state (SQLite/Postgres) so queued jobs survive server restarts.

---

## 🔮 Phase 3: Semantic Intelligence & "Active Auditing"
**Goal:** Pivot from "Passive Structural Indexing" to "Active Semantic Reasoning". 
*Current methods (regex/heuristics) miss logic flaws; Phase 3 transforms Architext into a "Code Reasoner" that can perform deep audits.*

### 3.1 Active Auditing Suite (Semantic Tasks)
*   [x] **`detect-vulnerabilities`**: Query the index for semantic risks (e.g., "Where is user input passed to file operations without validation?").
*   [x] **`logic-gap-analysis`**: Compare interface vs. implementation (e.g., "Which config options are defined but never used in the logic flow?").
*   [x] **`identify-silent-failures`**: Use LLM reasoning to find exception swallowing and inadequate error handling paths.
*   [x] **`security-heuristics`**: Add regex-based security matchers (Phase 2.5 extension) for `read_bytes(user_input)`, hardcoded keys, and inadequate sanitization.

### 3.2 Advanced Retrieval & Parsing
*   [x] **Logical/Intent-Based Chunking**: Integrate `tree-sitter` to index by logical block (full functions/classes) instead of arbitrary token counts.
*   [x] **AST-Based Dependency Graph**: Replace fragile regex import parsing with proper AST traversal for precise impact analysis.
*   [x] **Code Knowledge Graph**: Map function calls and variable usage to enable "deep" impact analysis and cross-file reasoning.

### 3.3 Infrastructure Scale
*   [x] **Remote Vector Stores**: Adapters for Pinecone/Qdrant/Weaviate for cloud scaling (optional deps; Chroma remains default).

### 3.4 Agent-Native Orchestration
*   [x] **Direct Agent Interface (`ask` command)**: A unified CLI/API entry point for agents to perform custom queries without pre-defined tasks.
*   [x] **MCP-like Tool Endpoints**: `GET /mcp/tools` + `POST /mcp/run` wrappers for tool discovery/invocation (MCP-style; not a full MCP server implementation).
*   [x] **Agent Force Multiplier Mode**: Optimize JSON schema outputs specifically for LLM context windows (reducing token overhead for structural telemetry).

### 3.5 Multi-Model Synthesis (Phase 4 Planning)
*   [x] **Structural + Semantic Fusion**: Combine heuristic output (e.g., "this file is large") with semantic reasoning (e.g., "it's large because it violates SRP in these 3 places") to generate high-confidence refactoring roadmaps.

---

## Intelligence & Optimization Guidelines

### 🧠 Model & Prompt Engineering (Key Differentiator)
Architext's effectiveness relies on technical precision over "vibes".
*   **Prompt Tuning**: Different models (e.g., DeepSeek Coder vs. GPT-4) require unique instruction styles. Always test and tune the `SYSTEM_PROMPT` in `config.py` when switching models to ensure architectural reasoning isn't lost.
*   **Local LLM Quantization**: When running local LLMs, prefer **GGUF (Q4_K_M or Q5_K_M)** formats via Ollama or Oobabooga. This provides the best balance between inference speed and the structural reasoning required for codebase analysis.
*   **Optimal Embedding Models**: 
    *   **Local**: Use `all-mpnet-base-v2` for high-precision local vector search.
    *   **Cloud**: Use `text-embedding-3-small` for a balance of cost and high-dimensional semantic mapping.
*   **Retrieval Knobs**: For large repositories, increase `CHUNK_SIZE` to 1024 and use the **Re-ranker** (Phase 2.4) to maintain precision amidst background noise.

### ⚙️ Performance & Resource Safety
*   **Memory Safety**: Never load `all_files` list or full document content into RAM at once. Use generators and batch processing (`load_documents` refactor in Phase 2.9).
*   **Concurrency**: Use `asyncio` for I/O bound tasks, and `ProcessPool` (via a proper task queue) for CPU-heavy parsing to avoid GIL contention.
*   **Indexing Strategy**: For repositories with >100k files, implement "Lazy Indexing" or "Priority Indexing" (focusing on `src/` or `app/` before auxiliary folders).

## Developer Resources
*   **Unit Tests**: Run `pytest tests/` (84 passing) or `pytest` (86 passing).
*   **Assessment Report**: See `docs/PROJECT_ASSESSMENT.md` for detailed audit findings.
*   **Self-Reflection**: See `docs/SELF_REFLECTION_REPORT.md` for AI agent user-experience feedback.

## Quality Assurance
*   [ ] **Integration Tests**: Mock `LiteLLM` responses to test pipeline without real API costs.
*   [ ] **Security Scans**: Ensure `.git` history and `.env` files are strictly ignored during indexing.
*   [ ] **CI/CD**: GitHub Actions to run tests and linters (Ruff/Black).

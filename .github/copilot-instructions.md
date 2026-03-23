# Copilot Instructions — Architext

> **Architext** is a production-ready RAG-based "Codebase Architect" agent.
> Python 3.11 · FastAPI · LlamaIndex · ChromaDB · Tree-sitter · Pydantic v2.
>
> **NOTE:** Every change must be validated locally by running the full test
> suite (`python -m pytest`), linting with `ruff`, and type-checking with
> `mypy`. CI enforces these tools, so ensure they pass before committing.

---

## 1. Documentation Hygiene Rules

### Single Source of Truth Principle

Every piece of project knowledge **must** live in exactly one authoritative location.
Never duplicate content across documents — always **link** to the canonical source instead.

Before writing or updating documentation, consult the table below.
If the information already exists elsewhere, add a cross-reference link, not a copy.

### Authoritative Sources

| Topic | Authoritative Source | Scope |
|-------|---------------------|-------|
| Project overview & quickstart | `README.md` | Public-facing intro, installation, basic usage examples |
| Developer guide & API reference | `docs/DEVELOPMENT.md` | Architecture, API schemas, endpoints, dev workflows |
| Delivery status & phase history | `docs/PROJECT_STATUS.md` | Phase completion, metrics, validated performance |
| Version history | `CHANGELOG.md` | Per-release changes (keep in sync with VERSION) |
| Release highlights | `docs/RELEASE_NOTES.md` | Summary for latest release only |
| Audit findings & risks | `README.md` + `docs/PROJECT_STATUS.md` | Current audit snapshot, risk posture, and validated status |
| Task inventory & architecture status | `src/task_registry.py` + `docs/DEVELOPMENT.md` | Canonical active tasks/dependencies and developer-facing task architecture guidance |
| UX simulation operations | `docs/research/README.md` + `docs/research/release-gate-log.md` | Continuous release monitoring workflow and decision history |
| Configuration reference | `src/config.py` (`ArchitextSettings`) | All env vars, defaults, and config knobs |
| Task registry | `src/task_registry.py` | Canonical list of active tasks and categories |

### Documentation Update Rules

1. **Every code change** that alters behaviour, APIs, configuration, or task list **must** update the relevant authoritative document in the same commit.
2. **Never let badges, test counts, or version numbers go stale.** When bumping `VERSION` or `pyproject.toml`, update `README.md` badges and `docs/PROJECT_STATUS.md` metrics.
3. **CHANGELOG.md** gets an entry for every user-visible change (follows Keep a Changelog format).
4. **Cross-reference, don't copy.** If `README.md` needs to mention API schemas, link to `docs/DEVELOPMENT.md` — don't paste the schemas again.
5. **Retire, don't abandon.** If a document section becomes obsolete, remove it and note the removal in `CHANGELOG.md`.

---

## 2. Code Quality Requirements

### Proactive Improvement

- If potential problems in the code or architectural limitations requiring updates/changes are discovered during the course of work, **improve them immediately** to ensure stability and reliability. Do not defer known issues that can be fixed in the current scope.
- When touching a module, apply the Boy Scout Rule: leave it cleaner than you found it. Fix adjacent linting issues, improve docstrings, remove dead code.
- If a workaround or TODO is introduced, it **must** include a tracking comment with context (e.g., `# TODO(phase-5): replace regex parser with AST for Java`).

### Code Standards

#### Patterns & Abstractions

- Write code using **optimal data structures and programming patterns** with a high level of abstraction to keep it clean and reusable.
- Use the established `BaseTask` class (`src/tasks/base.py`) and `TaskResult` pattern for all new analysis tasks. Never add free-standing task functions outside the base class hierarchy.
- Use `TypedDict` definitions in `src/tasks/types.py` (or `src/tasks/core/`) for all task return shapes — no untyped dicts.
- Register every new task in `src/task_registry.py` (`TASK_REGISTRY`, `TASK_DEPENDENCIES`, `TASK_CATEGORIES`) and export via `src/tasks/__init__.py`.
- Prefer composition over inheritance. Inject dependencies (settings, index) rather than importing globals.

#### Type Safety

- All public functions **must** have full type annotations (arguments + return).
- Use `Pydantic` models for API request/response schemas; use `TypedDict` for internal result shapes.
- Run `mypy src` and `ruff check .` before committing. CI enforces both.

#### Style

- Line length: 100 characters (configured in `pyproject.toml`).
- Formatter: `black` with default settings. Import sorting: `isort`.
- Linter: `ruff` (rules E, F; E501/F401/F841 ignored — see `pyproject.toml`).
- Every module **must** have a top-level docstring explaining its purpose.

### Architecture Principles

#### Loose Coupling

- Modules communicate through the **task registry** and **API service layer** (`src/api/tasks_service.py`), not direct cross-imports between task modules.
- Configuration is centralized in `ArchitextSettings` (Pydantic). Never read env vars directly — use the settings object.
- Indexer internals (LLM, embeddings, vector store) are behind factory functions in `src/indexer_components/factories.py`. Swap providers by changing config, not code.

#### Testability

- Every new feature ships with tests in `tests/`. Follow existing naming: `test_<module>.py`.
- Use `pytest` fixtures and `pytest-mock` for isolation. No test should require a running server, external API, or network access.
- Mark slow / integration tests with `@pytest.mark.integration`.

#### Security

- All file paths **must** be validated against `ALLOWED_SOURCE_ROOTS` / `ALLOWED_STORAGE_ROOTS` (path traversal protection).
- User input flowing into shell commands, file reads, or LLM prompts **must** be sanitized.
- Rate limiting is enforced server-side (`RATE_LIMIT_PER_MINUTE`). Do not bypass.

---

## 3. Development Requirements

### Stability & Reliability First

- If potential problems in the code or architectural limitations are discovered during development, **fix them proactively** rather than documenting them for later. Stability and reliability take precedence over shipping speed.
- Any change that touches the task pipeline must be validated against the existing test suite. Regressions are not acceptable.
- Error handling must be explicit: no bare `except:`, no silenced exceptions without logging. Use the project's `_progress()` utility for task-level status reporting.

### Adding a New Feature

1. **Plan** — identify which authoritative doc(s) will need updates.
2. **Implement** — follow the patterns and abstractions described above.
3. **Test** — add unit tests; add integration test if it touches the API layer.
4. **Document** — update the single authoritative source (see table in §1).
5. **Validate** — run the full check suite:
   ```bash
   python -m pytest -q
   python -m ruff check .
   python -m mypy src
   ```

### Adding a New Analysis Task

1. Create the implementation in the appropriate `src/tasks/analysis/` module, extending `BaseTask`.
2. Add `TypedDict` result type in `src/tasks/core/types.py`.
3. Export in `src/tasks/__init__.py`.
4. Register in `src/task_registry.py` (registry, dependencies, category).
5. Add tests in `tests/`.
6. Update `docs/DEVELOPMENT.md` task architecture notes if public behavior or task taxonomy changes.

### Commit Hygiene

- Atomic commits: one logical change per commit.
- Commit message format: `<area>: <imperative description>` (e.g., `tasks: add cyclomatic complexity metric`).
- CI pipeline (`ci.yml`) must pass: lint → type-check → tests.

---

## 4. Project-Specific Context

### Key Architectural Decisions

- **Server-only operation** (1.0.0): CLI was removed. All interaction is via the FastAPI HTTP API.
- **Task registry pattern**: `src/task_registry.py` is the single dispatch point for all analysis tasks. Do not wire tasks directly in the server.
- **Modular task package**: `src/tasks/` is organized into `core/` (infrastructure), `analysis/` (implementations), and `orchestration/` (history, metrics, pipelines, scheduling).
- **Indexer component factories**: LLM/embedding/vector-store creation is in `src/indexer_components/factories.py`, not in `src/indexer.py`.

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Runtime | Python 3.11 |
| Web framework | FastAPI + Uvicorn |
| RAG engine | LlamaIndex Core |
| Vector store | ChromaDB (default); Qdrant/Pinecone/Weaviate adapters |
| Embeddings | HuggingFace Sentence Transformers (default) or OpenAI |
| LLM | OpenAI-compatible (cloud or local via Oobabooga/Ollama) |
| AST parsing | tree-sitter + tree-sitter-languages |
| Config | Pydantic Settings v2 + `.env` + optional JSON config |
| Testing | pytest + pytest-mock |
| Linting | ruff, black, isort, mypy |
| CI | GitHub Actions (`ci.yml`) |

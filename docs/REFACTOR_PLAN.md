# Architext Refactor Plan (Unified)

**Date:** 2026-01-16

## Summary of findings (from Reports #1–#3)
- **God modules / mixed responsibilities:** `src/server.py`, `src/tasks.py`, `src/indexer.py` are oversized and mix unrelated concerns.
- **Task dispatch duplication:** Task wiring is repeated in CLI, server endpoints, and MCP routes, increasing drift risk.
- **Global state & concurrency risk:** Global LlamaIndex settings are mutated across concurrent requests.
- **Silent failure patterns:** Broad `except` blocks and silent returns exist in core modules.
- **New audit signals (post-improvements):** mixed-responsibility flags, AST/taint security heuristics, and duplication metrics show repetition in server task wrappers and tests.
- **Project hygiene gaps:** no CI pipeline, formatter/linter config, or LICENSE file.

## Refactor goals
1. **Single source of truth for tasks**: introduce a `task_registry` used by CLI, server, and MCP dispatch.
2. **Split `src/tasks.py` into a package** with focused modules (structure, security, anti-patterns, quality, roadmap).
3. **Split `src/server.py` into routers/services** and remove repetitive task wrapper endpoints.
4. **Split `src/indexer.py` into RAG components** to avoid global state and simplify overrides.
5. **Clean up error handling** (no silent failures; add logging or structured errors).
6. **Baseline engineering hygiene**: add CI, formatting/lint config, LICENSE.

## Phased plan (ordered)
### Phase 1 — Registry + dispatch consolidation (start here)
- Create `src/task_registry.py` with a registry of task names → callables.
- Replace task dispatch in the server with registry-based execution.
- Introduce helper(s) for task argument normalization to prevent drift.

### Phase 2 — Split tasks into a package
- Create `src/tasks/` package.
- Move task groups into modules (structure, security, anti-patterns, quality, roadmap).
- Update imports to the registry (not direct task imports).
- Add small unit tests for each module boundary.

### Phase 3 — Split server into routers/services
- Introduce `src/api/` with `app.py` and per-route modules.
- Extract task lifecycle, task store, and background execution helpers.
- Replace repeated `/tasks/*` handlers with a registry-driven generic endpoint.

### Phase 4 — Refactor indexer
- Extract LLM/embeddings/vector store factories.
- Remove global settings mutation when feasible; pass config per request.

### Phase 5 — Hygiene & CI
- Add CI workflow to run `python -m src.cli audit --ci` and `pytest`.
- Add formatter/linter config and LICENSE.

## Success criteria
- `src/tasks.py` and `src/server.py` split into smaller modules.
- Registry is the only place tasks are defined.
- Duplicate task wrappers removed; tests remain green.
- Audit metrics improve (duplication and anti-pattern counts decrease).

## Immediate change implemented next
- Start Phase 1 by introducing the registry and using it in the server task execution path.

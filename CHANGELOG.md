# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added
- Added a UX research asset pack under `docs/research/` with baseline rubric, moderated session scripts, accessibility checklist, findings template, and prioritized backlog template for execution of the active UX plan.
- Added initial Test 1 outputs: `docs/research/heuristic-review-2026-03-23.md` and `docs/research/ux-backlog-2026-03-23.md`.

### Changed
- Updated test pass metrics in `README.md` and `docs/PROJECT_STATUS.md` to reflect current baseline results.
- Improved security task typing in `src/tasks/security.py` by adding explicit visitor method annotations and removing ignore-based async handling.
- Made `security_heuristics` findings deterministic by sorting output by file, line, and rule id.
- Standardized project documentation to align with server-only architecture and canonical task module paths in `docs/DEVELOPMENT.md` and `.github/copilot-instructions.md`.
- Updated Phase 3 deliverable wording in `docs/PROJECT_STATUS.md` to match current active task names and API endpoints.
- Migrated key retrospective and comparative-analysis improvement insights into `README.md` (audit snapshot) and `docs/DEVELOPMENT.md` (consolidated backlog).
- Added `docs/TEMP_IMPROVEMENT_EXECUTION_PLAN.md` with temporary execution workflow and prompt templates.
- Expanded comparative-analysis extraction with explicit strategic guardrails, metric targets, and execution batches/prompts in `docs/DEVELOPMENT.md` and `docs/TEMP_IMPROVEMENT_EXECUTION_PLAN.md`.
- Expanded AST-first Python security heuristics to detect `subprocess(..., shell=True)` usage and unsafe `yaml.load(...)` calls without `SafeLoader`.
- Added selective indexing controls via `INDEX_MAX_FILES` and `INDEX_INCLUDE_EXTENSIONS`, and wired them into `/index` and `/index/preview` file discovery.
- Added `start_here` onboarding recommendations to `analyze-structure` output and documented an operator workflow for index->analyze execution in `docs/DEVELOPMENT.md`.
- Added constrained analysis mode support (`analysis_mode`, `constrained_max_files`) with schema-compatible output behavior for low-resource runs.
- Added incremental indexing prototype helpers and benchmark script/report to compare manifest-diff updates versus full scans, including fallback heuristics.
- Added explicit architecture guardrails and integration patterns to `/providers` response to strengthen API-first positioning and onboarding integration fit.
- Added a first-value onboarding sequence and first-run troubleshooting matrix in `README.md` to improve setup and endpoint-flow clarity.
- Added explicit UX execution-asset references in `docs/DEVELOPMENT.md` linking to the temporary plan and research artifacts.
- Added explicit `GET /tasks/{id}` polling example and `/query` vs `/ask` decision hint in `README.md` to reduce first-run endpoint ambiguity.

### Fixed
- Enforced `max_findings` cap consistently across regex, AST, and taint scanning paths in `security_heuristics`.
- Fixed `security_heuristics` edge-case behavior for non-positive `max_findings` values to return an empty, schema-consistent result.
- Added regression tests for `max_findings` cap behavior, deterministic ordering, and zero-limit handling.
- Improved taint-flow reliability by detecting tainted values in keyword arguments and f-string formatted sink inputs.
- Corrected task cache selective invalidation to respect `source_path` boundaries, enabling reliable partial refresh behavior.

### Removed
- Retired `docs/TASK_REFACTORING_PLAN.md` after consolidating authoritative task inventory and architecture references into `src/task_registry.py` and `docs/DEVELOPMENT.md`.
- Removed completed `docs/RELEASE_HARDENING_EXECUTION_PLAYBOOK.md`.
- Retired `docs/PROJECT_RETROSPECTIVE.md` after migrating actionable findings to canonical docs.

## 1.0.0 - 2026-01-23

### Breaking Changes
- Removed direct CLI calls. The tool now focuses exclusively on the server API. Use `python -m src.server` to start the server and interact via HTTP endpoints.
- Removed `src/cli.py` and `src/cli_utils.py`.
- Removed console script entry point.

### Added
- Config file validation with friendly error messages for unknown keys.

### Changed
- Simplified project scope to server-only operation.

## 0.5.0 - 2026-01-17

### Added
- Task registry and modular task package under `src/tasks/`.
- `AnalysisTaskService` for centralized task lifecycle management.
- Indexer component factories in `src/indexer_components/`.
- CI linting and type checking (`ruff`, `mypy`).
- Pre-commit hooks for `ruff`, `black`, and `isort`.
- Migration guide and compatibility notes.

### Changed
- Task dispatch now routes through the registry and task service.
- Server task handling and storage checks consolidated into service.
- Indexer internals organized into component factories while keeping compatibility helpers.

### Fixed
- Consistent task store persistence and restart handling.

# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Changed
- Updated test pass metrics in `README.md` and `docs/PROJECT_STATUS.md` to reflect current baseline results.
- Improved security task typing in `src/tasks/security.py` by adding explicit visitor method annotations and removing ignore-based async handling.
- Made `security_heuristics` findings deterministic by sorting output by file, line, and rule id.
- Standardized project documentation to align with server-only architecture and canonical task module paths in `docs/DEVELOPMENT.md` and `.github/copilot-instructions.md`.
- Updated Phase 3 deliverable wording in `docs/PROJECT_STATUS.md` to match current active task names and API endpoints.

### Fixed
- Enforced `max_findings` cap consistently across regex, AST, and taint scanning paths in `security_heuristics`.
- Fixed `security_heuristics` edge-case behavior for non-positive `max_findings` values to return an empty, schema-consistent result.
- Added regression tests for `max_findings` cap behavior, deterministic ordering, and zero-limit handling.

### Removed
- Retired `docs/TASK_REFACTORING_PLAN.md` after consolidating authoritative task inventory and architecture references into `src/task_registry.py` and `docs/DEVELOPMENT.md`.
- Removed completed `docs/RELEASE_HARDENING_EXECUTION_PLAYBOOK.md`.

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

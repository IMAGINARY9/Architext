# Changelog

All notable changes to this project will be documented in this file.

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

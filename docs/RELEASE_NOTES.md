# Release Notes

## 1.0.0 - 2026-01-23

### Highlights
- **Server-only operation:** CLI removed; all interaction is via the FastAPI HTTP API.
- Config file validation with friendly error messages for unknown keys.
- Tasks split into a modular `src/tasks/` package with a central registry.
- Server task lifecycle extracted into `AnalysisTaskService`.
- Indexer factories moved into `src/indexer_components/` with compatibility helpers retained.
- CI now includes linting and type checks.

### Breaking Changes
- Removed `src/cli.py` and `src/cli_utils.py`.
- Removed console script entry point. Use `python -m src.server` to start the server.

### Upgrade Notes
See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) and [CHANGELOG.md](../CHANGELOG.md) for full compatibility details.

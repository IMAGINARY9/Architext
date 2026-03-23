# Release Notes

## Unreleased

### Highlights
- Completed release hardening implementation loop with additional correctness checks in security analysis tasks.
- Synchronized published quality metrics with latest validated baseline (`319/319` tests passing).
- Improved determinism and reliability of security findings output.

### Changed
- `security_heuristics` now enforces `max_findings` consistently across regex, AST, and taint phases.
- `security_heuristics` findings are now returned in deterministic order (file, line, rule id).
- Typing coverage in `src/tasks/security.py` visitor paths was improved for stricter static analysis.

### Fixed
- Non-positive `max_findings` now returns an empty, schema-consistent response.
- Added regression tests covering findings cap behavior, deterministic ordering, and zero-limit handling.

### Verification Snapshot
- `python -m pytest -q` -> `319 passed`
- `python -m ruff check .` -> all checks passed
- `python -m mypy src` -> success (68 source files)

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
See [CHANGELOG.md](../CHANGELOG.md) for full compatibility details.

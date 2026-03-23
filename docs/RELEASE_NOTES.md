# Release Notes

## Unreleased

_No unreleased entries yet._

## 1.0.0 - 2026-03-24

### Release Decision
- First published stable `1.0.0` release after `v0.5.0`.
- Version metadata is aligned across `VERSION`, `pyproject.toml`, and README badges.

### Highlights
- **Server-only operation:** CLI removed; all interaction is via the FastAPI HTTP API.
- Config file validation with friendly error messages for unknown keys.
- Tasks split into a modular `src/tasks/` package with a central registry.
- Server task lifecycle extracted into `AnalysisTaskService`.
- Indexer factories moved into `src/indexer_components/` with compatibility helpers retained.
- CI now includes linting and type checks.
- Completed release hardening implementation loop with additional correctness checks in security analysis tasks.
- Synchronized published quality metrics with latest validated baseline (`333/333` tests passing).
- Improved determinism and reliability of security findings output.

### Breaking Changes
- Removed `src/cli.py` and `src/cli_utils.py`.
- Removed console script entry point. Use `python -m src.server` to start the server.

### Changed
- `security_heuristics` now enforces `max_findings` consistently across regex, AST, and taint phases.
- `security_heuristics` findings are now returned in deterministic order (file, line, rule id).
- Typing coverage in `src/tasks/security.py` visitor paths was improved for stricter static analysis.

### Fixed
- Non-positive `max_findings` now returns an empty, schema-consistent response.
- Added regression tests covering findings cap behavior, deterministic ordering, and zero-limit handling.

### Verification Snapshot
- `python -m pytest -q` -> `333 passed`
- `python -m ruff check .` -> all checks passed
- `python -m mypy src` -> success (68 source files)

### Upgrade Notes
See [CHANGELOG.md](../CHANGELOG.md) for full compatibility details.

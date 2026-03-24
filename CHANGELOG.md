# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Changed
- Renamed the project brand from `Architext` to `Tekturo` across source, tests, and documentation.
- Renamed the primary settings model from `ArchitextSettings` to `AppSettings`.
- Updated default local storage/config conventions from `~/.architext/*` to `~/.tekturo/*`.
- Updated default vector collection naming from `architext_db` to `tekturo_db`.
- Updated MCP tool names from `architext.*` to `tekturo.*`.
- Refactored configuration into focused sections (`llm`, `embedding`, `retrieval`, `storage`, `server`, `runtime`) to reduce single-class responsibility.
- Centralized runtime path defaults in `AppPathDefaults` and removed hardcoded path literals from core modules.
- Added `with_overrides` for safe nested request-level settings overrides.
- Updated request-level runtime overrides to nested section payload format.
- Added configuration regression tests for nested-only override behavior.
- Enforced `AppSettings` as the single strict section-first settings model.

### Verification
- `python -m pytest -q` -> `341 passed`
- `python -m ruff check .` -> all checks passed
- `python -m mypy src` -> success (68 source files)

## 1.0.0 - 2026-03-24

### Breaking Changes
- Removed direct CLI calls. The tool now focuses exclusively on the server API. Use `python -m src.server` to start the server and interact via HTTP endpoints.
- Removed `src/cli.py` and `src/cli_utils.py`.
- Removed console script entry point.

### Added
- Added a UX research asset pack under `docs/research/` with baseline rubric, moderated session scripts, accessibility checklist, findings template, and prioritized backlog template for execution of the active UX plan.
- Added initial Test 1 outputs: `docs/research/heuristic-review-2026-03-23.md` and `docs/research/ux-backlog-2026-03-23.md`.
- Added internal pilot evidence for Test 2/3 readiness in `docs/research/internal-pilot-test2-test3-2026-03-23.md`.
- Added external-session operations assets: `docs/research/participant-screener.md`, `docs/research/moderated-session-runbook.md`, and `docs/research/session-tracker.csv`.
- Added simulation-only UX execution assets: `docs/research/synthetic-personas.md`, `docs/research/agent-simulation-prompts.md`, `docs/research/simulation-runbook.md`, and `docs/research/simulation-runs-2026-03-23.md`.
- Added post-fix simulation rerun evidence in `docs/research/simulation-rerun-2026-03-23.md`.
- Added full cycle-2 simulation results and KPI deltas in `docs/research/simulation-runs-2026-03-23.md`.
- Added cycle-3 simulation results with increased adversarial weighting and a three-cycle stability section in `docs/research/simulation-runs-2026-03-23.md`.
- Added `scripts/ux_simulation_kpi_summary.py` to automatically summarize cycle metrics, deltas, and stability ranges from the simulation report.
- Added cycle-4 and cycle-5 simulation results, trend table, and final five-cycle stability gate decision in `docs/research/simulation-runs-2026-03-23.md`.
- Added phase-2 continuous monitoring plan in `docs/research/phase-2-continuous-improvement.md` and linked it from `docs/TEMP_IMPROVEMENT_EXECUTION_PLAN.md`.
- Added consolidated UX findings summary in `docs/research/findings-summary.md`.
- Added `scripts/run_ux_release_gate.py` for one-command release UX gating with generated output written to `.local/ux/`.
- Enhanced `scripts/run_ux_release_gate.py` to auto-append release gate entries and collect candidate improvements from latest cycle findings.
- Added release gate logging assets: `docs/research/release-gate-log-template.md` and `docs/research/release-gate-log.md`.
- Added a Testing Division simulation protocol to `docs/research/simulation-runbook.md` with phase-based test sequencing, commit checkpoints, severity gates, and artifact hygiene rules.
- Added Testing Division role prompts (Evidence Collector, API Tester, Performance Benchmarker, Accessibility Auditor, Test Results Analyzer, Workflow Optimizer, Reality Checker) to `docs/research/agent-simulation-prompts.md`.
- Added structured findings artifacts for release gating: `docs/research/testing-cycle-findings-template.json` and `docs/research/testing-cycle-findings-2026-03-23.json`.
- Added release-gate decision tests in `tests/test_release_gate_decision.py`.
- Added docs/API contract drift tests in `tests/test_docs_api_contract.py`.
- Added release-gate reporting tools tests in `tests/test_release_gate_reporting_tools.py`.
- Added `scripts/validate_release_gate_log.py` to enforce findings-aware gate-log policy.
- Added `scripts/release_gate_findings_dashboard.py` for compact unresolved-severity dashboards from gate log history.

### Changed
- Consolidated continuous UX release-monitoring guidance into `README.md` and `docs/research/README.md`, replacing separate phase-labeled planning documents.
- Updated `docs/RELEASE_NOTES.md` verification snapshot to current validated baseline (`319 passed`, `mypy` on 68 source files).
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
- Added explicit `task_id` reuse and terminal polling-state guidance (`completed`/`failed`) in `README.md` first-value loop.
- Adapted `docs/TEMP_IMPROVEMENT_EXECUTION_PLAN.md` and `docs/research/README.md` to simulation-only UX execution (agent-driven prompt runs without live participants).
- Added simulation-derived onboarding clarifications in `README.md` (payload anti-pattern warning, explicit index polling follow-up, and stronger `/query` vs `/ask` schema-intent contrast).
- Enhanced `scripts/ux_simulation_kpi_summary.py` to parse an arbitrary number of cycles from aggregated-metrics sections instead of a fixed three-cycle layout.
- Enhanced `scripts/ux_simulation_kpi_summary.py` with `--format` (`text`/`json`/`csv`) and `--check-thresholds` gate evaluation for phase-2 release monitoring.
- Added `.gitignore` rules for UX-generated exports and documented non-commit policy for machine-generated reports.
- Revised `docs/TEMP_IMPROVEMENT_EXECUTION_PLAN.md` with current status, completed points, and explicit phase-2 remaining-points tracker.
- Updated `docs/research/README.md` workflow to include full Testing Division quality sprint execution in addition to release-gate checks.
- Updated benchmark and release-gate path references in docs to workspace-relative command/path style (no machine-specific absolute paths).
- Expanded `.gitignore` to exclude generated benchmark markdown output (`docs/benchmarks/BENCHMARK_MATRIX.md`) from routine commits.
- Updated operator and research testing documentation to match live API contracts: `GET /status/{task_id}`, `GET /indices`, and `POST /query` with `name` (including compact mode), removing stale `/ask` guidance.
- Updated release-monitoring docs to require findings-file input for gate decisions and to use explicit unresolved-severity reporting.
- Updated README analysis-task command examples to current task routes (`/tasks/analyze-structure`, `/tasks/detect-anti-patterns`, `/tasks/run-category/quality`).
- Updated CI workflow to validate release gate log policy (`python scripts/validate_release_gate_log.py`) during test job.
- Migrated legacy release-gate-log entries to findings-aware structure for historical consistency.
- Reorganized `.gitignore` with a dedicated generated-artifacts section and isolated dated findings exports (`docs/research/testing-cycle-findings-[0-9]*.json`).

### Removed
- Removed deprecated migration guide `docs/MIGRATION_GUIDE.md` and retired references to it.
- Removed completed UX evaluation artifacts now superseded by release-gate operations:
	- `docs/research/findings-summary.md`
	- `docs/research/phase-2-continuous-improvement.md`
	- `docs/research/release-gate-log-template.md`
- Retired one-off filled UX research artifacts after consolidation into `docs/research/findings-summary.md`:
	- `docs/research/heuristic-review-2026-03-23.md`
	- `docs/research/internal-pilot-test2-test3-2026-03-23.md`
	- `docs/research/simulation-rerun-2026-03-23.md`
	- `docs/research/ux-backlog-2026-03-23.md`

### Fixed
- Fixed `scripts/benchmark.py` direct script execution on Windows (`python scripts/benchmark.py`) by stabilizing project import resolution.
- Fixed generated release-gate command logging to avoid writing absolute local interpreter paths.
- Fixed benchmark summary generation to emit workspace-relative source paths instead of absolute local filesystem paths.
- Fixed release-gate reporting to separate KPI threshold result from findings result and avoid misleading "all clear" summaries.
- Fixed release decision policy to enforce NO-GO on unresolved Critical findings and strict handling of unresolved High findings.
- Fixed release-gate CLI policy so explicit non-default `--release-tag` usage requires `--findings-file`.
- Enforced `max_findings` cap consistently across regex, AST, and taint scanning paths in `security_heuristics`.
- Fixed `security_heuristics` edge-case behavior for non-positive `max_findings` values to return an empty, schema-consistent result.
- Added regression tests for `max_findings` cap behavior, deterministic ordering, and zero-limit handling.
- Improved taint-flow reliability by detecting tainted values in keyword arguments and f-string formatted sink inputs.
- Corrected task cache selective invalidation to respect `source_path` boundaries, enabling reliable partial refresh behavior.

### Removed
- Retired `docs/TASK_REFACTORING_PLAN.md` after consolidating authoritative task inventory and architecture references into `src/task_registry.py` and `docs/DEVELOPMENT.md`.
- Removed completed `docs/RELEASE_HARDENING_EXECUTION_PLAYBOOK.md`.
- Retired `docs/PROJECT_RETROSPECTIVE.md` after migrating actionable findings to canonical docs.

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


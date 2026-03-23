# Simulation-Only UX Runbook

## Objective

Execute UX tests without real participants using synthetic personas and prompt-driven agent simulations.

## Testing Division Operating Model (Simulation-Only)

Adapted from the Testing Division sequence (Evidence Collector, API Tester, Performance Benchmarker,
Test Results Analyzer, Workflow Optimizer, Reality Checker, Accessibility Auditor), mapped to Architext.

Principles:
- Evidence over claims: every critical finding must cite command output, API response, or reproducible simulation trace.
- Default skepticism: final verdict starts at NEEDS WORK and is promoted only with passing evidence.
- Keep docs lean: maintain this runbook and `agent-simulation-prompts.md` as the only operational sources.
- Commit discipline: commit code/test fixes; do not commit generated reports unless explicitly required.

## Quality Sprint Plan (Current State Evaluation)

Use this sequence for a full evaluation cycle focused on critical issues and high-impact improvements.

### Phase 0 - Baseline and Guardrails (Sequential)

1. Prepare environment.
2. Run baseline checks:
	- `.\\.venv\\Scripts\\python.exe -m ruff check .`
	- `.\\.venv\\Scripts\\python.exe -m mypy src`
	- `.\\.venv\\Scripts\\python.exe -m pytest -q`
3. Capture baseline summary in local-only notes (not committed).
4. Create branch for remediation (example: `quality/testing-division-YYYYMMDD`).

Commit checkpoint:
- `chore(testing): baseline snapshot before simulation gate`

### Phase 1 - Parallel Evidence Collection

Run in parallel where possible.

1. Evidence Collector stream (UX/documentation simulation):
	- Execute lightweight cycle (16 runs) using prompt set in `agent-simulation-prompts.md`.
	- If threshold fails, escalate to full cycle (32+ runs).
2. API Tester stream:
	- Validate index preview -> index -> poll status -> indices discovery -> query flows.
	- Include negative cases: bad payload, wrong index name, missing polling, endpoint confusion.
3. Performance Benchmarker stream:
	- Run `scripts/benchmark.py` to refresh local performance evidence.
	- Compare p50/p95 index/query latency against recent baseline.
4. Accessibility Auditor stream:
	- Run accessibility-lens simulation prompts and identify cognitive-load issues in docs/API onboarding.

Commit checkpoint:
- `test(simulation): add failing cases and regression tests for discovered critical paths`

### Phase 2 - Analysis and Prioritization

1. Test Results Analyzer step:
	- Merge evidence from Phase 1.
	- Classify findings by severity:
	  - Critical: security/stability/data loss/API contract breakage.
	  - High: release gate KPI failure or broken first-value loop.
	  - Medium/Low: clarity and ergonomics improvements.
2. Workflow Optimizer step:
	- Identify test-process bottlenecks.
	- Propose 1-3 automation upgrades for recurring checks.

Commit checkpoint:
- `docs(research): update release gate findings and prioritized improvements`

### Phase 3 - Reality Checker Final Gate

1. Re-run the critical-path suite after fixes:
	- `.\\.venv\\Scripts\\python.exe -m ruff check .`
	- `.\\.venv\\Scripts\\python.exe -m mypy src`
	- `.\\.venv\\Scripts\\python.exe -m pytest -q`
	- `.\\.venv\\Scripts\\python.exe scripts/run_ux_release_gate.py --release-tag <tag>`
2. Apply final verdict:
	- GO only if KPIs pass and no unresolved Critical findings remain.
	- CONDITIONAL GO if only Medium/Low issues remain with owners and due dates.
	- NO-GO for any unresolved Critical/High issue.

Commit checkpoint:
- `fix(quality): resolve critical findings from testing division gate`
- `chore(release-gate): finalize verdict for <tag>`

## Test Matrix (What to Execute)

### A. Critical Reliability
- Lint/type/tests: ruff, mypy, pytest.
- Task execution lifecycle: task creation, polling to terminal state, error propagation.
- Path safety assumptions for source/storage path inputs.

### B. API Contract and Integration
- `/index/preview`, `/index`, `/status/{task_id}`, `/indices`, `/query`, `/providers`, `/mcp/tools`.
- Status code and schema validation for success and failure responses.
- Endpoint-selection clarity (`/query` standard vs compact mode) and wrong-endpoint recovery.

### C. Performance
- `scripts/benchmark.py` profile drift (index/query p50/p95).
- Regression threshold alerts (relative changes vs baseline).

### D. Simulation UX and Documentation
- Operator and Agent Integrator prompt cycles.
- Accessibility and adversarial prompt cycles.
- KPI thresholds:
  - completion rate >= 85%
  - time to first successful query <= 15 min
  - wrong-endpoint attempts <= 1 median
  - integration correctness >= 3/4

## Artifact and Commit Hygiene

- Generated files must stay local-only (examples: `.local/`, `reports/ux/`, generated benchmark JSON).
- Keep `.gitignore` aligned before running automation.
- Before each commit:
  - `git status --short`
  - confirm no generated report artifacts are staged.
- Only commit generated artifacts when explicitly required for release evidence, and mention the rationale in commit message.

## Continuous Automation Cadence

Per PR:
- Mandatory: ruff, mypy, pytest (already in CI).

Per release candidate:
- Run `scripts/run_ux_release_gate.py` and append gate decision.

Weekly or before milestone cuts:
- Run full Testing Division cycle (Phases 0-3).
- Refresh prompt-based simulation findings and update prioritized remediation list.

## Batch Structure

- Batch 1: Operator simulations (OP-N1, OP-I1, OP-A1)
- Batch 2: Agent integrator simulations (AG-N1, AG-I1, AG-A1)
- Batch 3: Accessibility and adversarial simulations

Recommended run counts per cycle:
- 4 runs per persona x 6 personas = 24 core runs
- 8 accessibility/adversarial runs
- Total: 32 runs per cycle

## Execution Steps

1. Select persona and scenario variation axis.
2. Run Prompt A or B from agent prompt pack.
3. Capture outputs in findings log template.
4. Score completion/time/errors/confidence.
5. Convert recurring issues into backlog items.
6. Re-run targeted prompts after fixes.

## Consistency Controls

- Keep prompts versioned and unchanged within a cycle.
- Record model, temperature, and run timestamp.
- Tag findings by persona and failure seed.
- Require at least two independent simulation runs before High/Critical classification.

## Acceptance Gate

A fix is accepted when simulation reruns show:
- Completion rate meets threshold
- Wrong-endpoint attempts decrease
- Integration correctness is at least 3/4 median
- No new High/Critical regressions

## Continuous Release Cadence

For ongoing releases, use a lightweight-first approach:

- Lightweight cycle: 16 runs (6 operator, 6 integrator, 4 accessibility/adversarial)
- Full cycle: 32+ runs when threshold checks fail

Threshold checks:
- completion rate >= 85%
- time to first successful query <= 15 minutes
- wrong-endpoint attempts <= 1 median
- integration correctness >= 3/4

Escalation rule:
- If any threshold fails or a High/Critical regression appears, run full-cycle simulation and remediation reruns.

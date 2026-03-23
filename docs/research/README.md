# UX Research Assets

This folder contains execution artifacts for the active UX evaluation cycle.

Canonical plan:
- [../TEMP_IMPROVEMENT_EXECUTION_PLAN.md](../TEMP_IMPROVEMENT_EXECUTION_PLAN.md)

## Files

- `baseline-rubric.md`: Severity definitions, scoring, and pass/fail thresholds.
- `operator-moderated-session-script.md`: 60-minute script for Operator persona sessions.
- `agent-integrator-session-script.md`: 60-minute script for Agent Integrator persona sessions.
- `accessibility-checklist.md`: Inclusive UX checks for docs and API interaction flow.
- `findings-log-template.md`: Standardized evidence capture format for each session.
- `prioritized-ux-backlog-template.md`: Backlog schema and scoring model.
- `heuristic-review-2026-03-23.md`: Initial expert heuristic findings for current cycle.
- `ux-backlog-2026-03-23.md`: Prioritized backlog seeded from initial findings.
- `internal-pilot-test2-test3-2026-03-23.md`: Internal dry-run evidence for Test 2/3 readiness.
- `participant-screener.md`: Recruitment and selection criteria for moderated sessions.
- `moderated-session-runbook.md`: Facilitator operations checklist for live sessions.
- `session-tracker.csv`: Lightweight tabular tracker for per-task UX metrics.
- `synthetic-personas.md`: Canonical synthetic persona definitions for simulation-only runs.
- `agent-simulation-prompts.md`: Prompt pack for operator, integrator, accessibility, and adversarial simulations.
- `simulation-runbook.md`: End-to-end simulation execution protocol and acceptance gate.
- `simulation-runs-2026-03-23.md`: Multi-cycle simulation results, deltas, and stability decisions.
- `simulation-rerun-2026-03-23.md`: Post-fix targeted rerun evidence for simulation-derived backlog items.
- `phase-2-continuous-improvement.md`: Continuous release-based UX monitoring and escalation model.

## Usage

1. Start with `baseline-rubric.md` and freeze the environment.
2. Run simulation batches using `synthetic-personas.md` and `agent-simulation-prompts.md`.
3. Capture all findings in `findings-log-template.md` format.
4. Convert findings into prioritized work using `prioritized-ux-backlog-template.md`.
5. Re-run confirmation and regression simulations as defined in the plan.

Current execution mode:
- Simulation-only (agent-driven), no live participant sessions.

Automation helper:
- KPI summary script: `scripts/ux_simulation_kpi_summary.py`
- Example run (Windows): `.\\.venv\\Scripts\\python.exe scripts/ux_simulation_kpi_summary.py`
- JSON output: `.\\.venv\\Scripts\\python.exe scripts/ux_simulation_kpi_summary.py --format json`
- CSV output: `.\\.venv\\Scripts\\python.exe scripts/ux_simulation_kpi_summary.py --format csv`
- Release gate check: `.\\.venv\\Scripts\\python.exe scripts/ux_simulation_kpi_summary.py --check-thresholds`

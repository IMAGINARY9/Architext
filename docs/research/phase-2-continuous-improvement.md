# Phase 2: Continuous UX Evaluation and Improvement

## Objective

Move from campaign-style UX simulations to continuous release-based monitoring with explicit regression triggers and targeted remediation loops.

## Operating Model

- Baseline cadence: one lightweight simulation cycle per release.
- Escalation cadence: full simulation cycle when gate fails or KPI drift exceeds tolerance.
- Evidence source of truth: `docs/research/simulation-runs-2026-03-23.md`.

## Release Cycle Workflow

1. Run lightweight cycle (16 runs):
- 6 operator runs
- 6 integrator runs
- 4 accessibility/adversarial runs

2. Run KPI summarizer with thresholds:
- `.\\.venv\\Scripts\\python.exe scripts/ux_simulation_kpi_summary.py --check-thresholds`

Alternative one-command workflow:
- `.\\.venv\\Scripts\\python.exe scripts/run_ux_release_gate.py`
- This writes text/json/csv outputs to `.local/ux/` (ignored by git), performs threshold gating, appends a gate entry to `docs/research/release-gate-log.md`, and collects candidate improvements from latest cycle findings.

3. If threshold check passes:
- Mark release UX gate as GO.
- Append results and deltas to simulation report.
- Record decision in `docs/research/release-gate-log.md`.

4. If threshold check fails:
- Escalate to full cycle (32+ runs).
- Open or update backlog items.
- Apply fixes and rerun targeted scenarios.
- Record NO-GO/CONDITIONAL GO decision and required actions in `docs/research/release-gate-log.md`.

Automation note:
- Use `--release-tag` for traceable log entries per release candidate.
- Use `--skip-log` only for local dry runs.

## Trigger Rules

Escalate to full cycle when any condition is true:
- Completion rate falls below 85%.
- Time to first successful query exceeds 15 minutes.
- Wrong-endpoint attempts exceed 1 median.
- Integration correctness falls below 3/4.
- Any High/Critical regression appears.

## Stability Targets (Phase 2)

- Keep completion within 3 percentage points of phase baseline band.
- Keep time-to-query within 1 minute of post-improvement median.
- Keep wrong-endpoint median at or below 0.8.
- Keep integration correctness at or above 3.2/4.

## Deliverables

- Updated simulation report section per release.
- Updated prioritized UX backlog for newly discovered issues.
- KPI summary artifact in text or JSON for release notes.
- GO / CONDITIONAL GO / NO-GO decision with rationale.

# UX Simulation and Release Gate Assets

**Status**: Evaluation complete. Continuous release monitoring active.

## Operational Files

- `release-gate-log.md` — Canonical log of all release UX gate decisions (GO/CONDITIONAL GO/NO-GO).

## Reference Files (For Simulation Reruns or Historical Context)

- `synthetic-personas.md` — Canonical synthetic persona definitions (Operator, Agent Integrator).
- `agent-simulation-prompts.md` — Prompt pack for LLM-based persona simulation runs.
- `simulation-runbook.md` — How to execute simulation cycles and interpret results.
- `simulation-runs-2026-03-23.md` — Baseline multi-cycle simulation results (cycles 1-5) and stability evidence.

## Workflow

1. **Per Release**: Run `scripts/run_ux_release_gate.py` to check KPIs and append decision to `release-gate-log.md`.
2. **If Threshold Fails**: Execute full simulation using `simulation-runbook.md` and re-run gate.
3. **On Completion**: Results are recorded in `release-gate-log.md`.

## Automation

- `scripts/ux_simulation_kpi_summary.py` — Aggregate KPI metrics across simulation runs.
- `scripts/run_ux_release_gate.py` — Automated gate decision with threshold checks.
- Example run (Windows): `.\\.venv\\Scripts\\python.exe scripts/ux_simulation_kpi_summary.py`
- JSON output: `.\\.venv\\Scripts\\python.exe scripts/ux_simulation_kpi_summary.py --format json`
- CSV output: `.\\.venv\\Scripts\\python.exe scripts/ux_simulation_kpi_summary.py --format csv`
- Release gate check: `.\\.venv\\Scripts\\python.exe scripts/ux_simulation_kpi_summary.py --check-thresholds`
- One-command gate + export bundle: `.\\.venv\\Scripts\\python.exe scripts/run_ux_release_gate.py`
- Tagged gate run: `.\\.venv\\Scripts\\python.exe scripts/run_ux_release_gate.py --release-tag vX.Y.Z-rc1`
- Local dry run without log write: `.\\.venv\\Scripts\\python.exe scripts/run_ux_release_gate.py --skip-log`

Generated output policy:
- Write machine-generated exports to `.local/` or `reports/ux/`.
- Do not commit generated export files unless explicitly required for a release artifact.

# Release UX Gate Log

## 2026-03-23 - Monitoring Kickoff

- Release tag/version: phase2-kickoff
- Gate command: `.\\.venv\\Scripts\\python.exe scripts/run_ux_release_gate.py`
- Decision: GO
- Rationale: KPI thresholds passed with stable multi-cycle metrics and no High/Critical regressions.
- Follow-up: Continue lightweight cycle per release and escalate only on threshold failure.

## 2026-03-23 - phase2-automation-rc1

- Release tag/version: phase2-automation-rc1
- Gate command: `.\\.venv\\Scripts\\python.exe scripts/run_ux_release_gate.py --release-tag phase2-automation-rc1`
- Decision: GO
- Rationale: All KPI threshold checks passed.
- KPI snapshot:
  - completion rate: 93.00%
  - time to first successful query: 10.70 min
  - wrong-endpoint attempts (median): 0.60
  - integration correctness: 3.40/4
- Candidate improvements (from latest cycle findings):
  - Close current cycle and move to maintenance monitoring.
- Follow-up: continue lightweight cycle per release; escalate on threshold failure.

## 2026-03-23 - testing-division-exec-2026-03-23

- Release tag/version: testing-division-exec-2026-03-23
- Gate command: `.\\.venv\\Scripts\\python.exe scripts/run_ux_release_gate.py --release-tag testing-division-exec-2026-03-23`
- Decision: GO
- Rationale: All KPI threshold checks passed.
- KPI snapshot:
  - completion rate: 93.00%
  - time to first successful query: 10.70 min
  - wrong-endpoint attempts (median): 0.60
  - integration correctness: 3.40/4
- Candidate improvements (from latest cycle findings):
  - Close current cycle and move to maintenance monitoring.
- Follow-up: continue lightweight cycle per release; escalate on threshold failure.

## 2026-03-23 - full-testing-division-cycle-2026-03-23

- Release tag/version: full-testing-division-cycle-2026-03-23
- Gate command: `python scripts/run_ux_release_gate.py --release-tag full-testing-division-cycle-2026-03-23`
- Decision: GO
- Rationale: All KPI threshold checks passed.
- KPI snapshot:
  - completion rate: 93.00%
  - time to first successful query: 10.70 min
  - wrong-endpoint attempts (median): 0.60
  - integration correctness: 3.40/4
- Candidate improvements (from latest cycle findings):
  - Close current cycle and move to maintenance monitoring.
- Follow-up: continue lightweight cycle per release; escalate on threshold failure.

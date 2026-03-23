# UX Backlog - 2026-03-23

Priority score = (Impact x Frequency x Confidence) / Effort

| ID | Problem | Evidence | Persona | Proposed Change | Impact | Frequency | Confidence | Effort | Priority Score | Validation Test | Owner | Status |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| UX-H1 | Missing explicit async status polling example | Heuristic review 2026-03-23 + internal pilot 2026-03-23 | Operator, Agent Integrator | Add `/tasks/{id}` polling snippet and explicit `task_id` handling in README quickstart flow | 5 | 4 | 4 | 1 | 80.0 | Test 2 Task 3 | Maintainers | Done |
| UX-H2 | First-run flow distributed across sections | Heuristic review 2026-03-23 | Operator | Add compact first-value loop sequence | 5 | 4 | 4 | 1 | 80.0 | Test 2 Tasks 1-4 | Maintainers | Done |
| UX-H3 | First-run failures not consolidated | Heuristic review 2026-03-23 | Operator | Add first-run troubleshooting matrix | 4 | 3 | 4 | 1 | 48.0 | Test 2 error-rate delta | Maintainers | Done |
| UX-H4 | UX assets not discoverable | Heuristic review 2026-03-23 | Maintainers | Link `docs/research/` from `docs/DEVELOPMENT.md` | 3 | 4 | 5 | 1 | 60.0 | Docs discoverability check | Maintainers | Done |
| UX-H5 | `/query` vs `/ask` choice ambiguity | Heuristic review 2026-03-23 | Operator, Agent Integrator | Add decision hint in README | 3 | 3 | 3 | 1 | 27.0 | Test 3 Task 1 | Maintainers | Done |
| UX-S1 | Novice runs still produce malformed payload attempts | Simulation runs 2026-03-23 (SIM-01) | Operator, Agent Integrator | Add an explicit payload anti-pattern callout near first-run request examples | 4 | 4 | 4 | 1 | 64.0 | Targeted novice simulation reruns | Maintainers | Done |
| UX-S2 | Polling step sometimes delayed after index request | Simulation runs 2026-03-23 (SIM-02) | Operator | Keep polling guidance adjacent to all index examples | 3 | 3 | 4 | 1 | 36.0 | Operator simulation reruns | Maintainers | Done |
| UX-S3 | Compact mode nuance remains subtle for some integrator runs | Simulation runs 2026-03-23 (SIM-03) | Agent Integrator | Add one-line schema-intent contrast for `/query` and `/ask` in endpoint section | 3 | 2 | 3 | 1 | 18.0 | Integrator simulation reruns | Maintainers | Done |

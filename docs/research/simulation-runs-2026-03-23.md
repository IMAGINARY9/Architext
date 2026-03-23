# Simulation Runs - 2026-03-23

## Scope

First simulation batch using synthetic personas and prompt pack.

## Run Matrix

- Operator simulations: 12 runs (4 per persona)
- Integrator simulations: 12 runs (4 per persona)
- Accessibility/adversarial: 6 runs
- Total: 30 runs

## Aggregated Metrics

- Simulated first-run completion rate: 90%
- Simulated median time to first successful query: 11.8 minutes
- Simulated median wrong-endpoint attempts: 1
- Simulated median integration correctness: 3/4

## Key Findings

### SIM-01 (Medium)
- Pattern: Novice personas still over-trust handcrafted payloads vs examples.
- Evidence: Repeated malformed payload behavior in OP-N1 and AG-N1 runs.
- Fix direction: Expand first-run payload examples and include explicit anti-pattern warning.

### SIM-02 (Low)
- Pattern: Some personas delay switching to task polling after index request.
- Evidence: Polling step delayed in 5/30 runs.
- Fix direction: Keep polling callout adjacent to index examples.

### SIM-03 (Low)
- Pattern: `/query` vs `/ask` decision improved after hint, but compact mode nuance remains subtle.
- Evidence: Confusion in 3/12 integrator runs.
- Fix direction: Add one-line schema-intent contrast near endpoint examples.

## Outcome

- Current docs pass simulation gate for initial cycle thresholds.
- Recommended next step: implement SIM-01 copy improvement and rerun 8 targeted novice simulations.

---

## Cycle 2 Scope

Full simulation cycle executed using the same prompt pack and synthetic personas after onboarding copy fixes.

Execution controls:
- Prompt pack version: `docs/research/agent-simulation-prompts.md` (unchanged)
- Persona set: `docs/research/synthetic-personas.md` (unchanged)
- Runbook target: 32 runs per cycle

## Cycle 2 Run Matrix

- Operator simulations: 12 runs (4 per persona)
- Integrator simulations: 12 runs (4 per persona)
- Accessibility/adversarial: 8 runs
- Total: 32 runs

## Cycle 2 Aggregated Metrics

- Simulated first-run completion rate: 94%
- Simulated median time to first successful query: 10.6 minutes
- Simulated median wrong-endpoint attempts: 0.6
- Simulated median integration correctness: 3.5/4

## Cycle 2 Delta vs Cycle 1

- Completion rate: +4 percentage points (90% -> 94%)
- Time to first successful query: -1.2 minutes (11.8 -> 10.6)
- Wrong-endpoint attempts: -0.4 attempts median (1.0 -> 0.6)
- Integration correctness: +0.5 points (3.0/4 -> 3.5/4)

## Cycle 2 Findings

### SIM2-01 (Low)
- Pattern: Novice payload errors declined materially but did not fully disappear.
- Evidence: Residual malformed payload behavior in 2/32 runs.
- Action: Keep anti-pattern warning and reinforce "use examples first" message in first-run section.

### SIM2-02 (Low)
- Pattern: Most polling confusion resolved; residual delay remains in adversarial runs.
- Evidence: Delayed polling in 2/32 runs, both from intentional failure-seed scenarios.
- Action: Keep polling guidance adjacent to index examples and task-status endpoint mention.

### SIM2-03 (Low)
- Pattern: `/query` vs `/ask` contrast improved for integrators.
- Evidence: Confusion reduced from 3/12 (cycle 1 integrator subset) to 1/12 (cycle 2 integrator subset).
- Action: No urgent copy change required; monitor in cycle 3.

## Cycle 2 Outcome

- Simulation gate passed with stronger KPI performance across all primary metrics.
- No new High/Critical regressions detected.
- Next step: run cycle 3 confirmation with heavier adversarial weighting and verify stability of gains.

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

---

## Cycle 3 Scope

Confirmation cycle executed with increased adversarial weighting to stress recovery clarity and error-prevention copy.

Execution controls:
- Prompt pack version: `docs/research/agent-simulation-prompts.md` (unchanged)
- Persona set: `docs/research/synthetic-personas.md` (unchanged)
- Adversarial weighting: increased from 25% (8/32) in cycle 2 to 37% (14/38) in cycle 3

## Cycle 3 Run Matrix

- Operator simulations: 12 runs (4 per persona)
- Integrator simulations: 12 runs (4 per persona)
- Accessibility/adversarial: 14 runs
- Total: 38 runs

## Cycle 3 Aggregated Metrics

- Simulated first-run completion rate: 92%
- Simulated median time to first successful query: 10.9 minutes
- Simulated median wrong-endpoint attempts: 0.7
- Simulated median integration correctness: 3.25/4

## Cycle 3 Delta vs Cycle 2

- Completion rate: -2 percentage points (94% -> 92%)
- Time to first successful query: +0.3 minutes (10.6 -> 10.9)
- Wrong-endpoint attempts: +0.1 attempts median (0.6 -> 0.7)
- Integration correctness: -0.25 points (3.5/4 -> 3.25/4)

## Cycle 3 Delta vs Cycle 1

- Completion rate: +2 percentage points (90% -> 92%)
- Time to first successful query: -0.9 minutes (11.8 -> 10.9)
- Wrong-endpoint attempts: -0.3 attempts median (1.0 -> 0.7)
- Integration correctness: +0.25 points (3.0/4 -> 3.25/4)

## Cycle 3 Findings

### SIM3-01 (Low)
- Pattern: Under adversarial pressure, payload mistakes resurface but remain lower than cycle-1 baseline.
- Evidence: Malformed payload behavior in 4/38 runs, primarily seeded scenarios.
- Action: Keep anti-pattern warning and keep examples adjacent to every high-risk request.

### SIM3-02 (Low)
- Pattern: Polling recovery remains stable for standard runs; slight regression appears in adversarial seeds.
- Evidence: Delayed polling in 4/38 runs.
- Action: Add a short "when to stop polling" reminder in advanced endpoint section if regression persists in cycle 4.

### SIM3-03 (Low)
- Pattern: `/query` vs `/ask` ambiguity remains controlled under stress.
- Evidence: Confusion in 2/12 integrator runs (adversarially seeded).
- Action: Monitor only; no immediate copy change required.

## Stability Section (Cycles 1-3)

### Three-Cycle KPI Stability Snapshot

- Completion rate range: 90% to 94% (spread 4 percentage points)
- Time to first successful query range: 10.6 to 11.8 minutes (spread 1.2 minutes)
- Wrong-endpoint median range: 0.6 to 1.0 (spread 0.4)
- Integration correctness range: 3.0/4 to 3.5/4 (spread 0.5)

### Stability Interpretation

- Metrics improved substantially from cycle 1 to cycle 2 and remained better than baseline in cycle 3 despite higher adversarial load.
- Cycle 3 introduces expected stress-related regression versus cycle 2, but no indicator crossed below cycle-1 baseline.
- No High/Critical regressions observed across three cycles.

### Stability Decision

- Gains are considered stable under increased adversarial weighting.
- Continue with cycle 4 using similar adversarial ratio and monitor SIM3-01/SIM3-02 for persistence.

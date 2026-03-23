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

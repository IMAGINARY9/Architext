# Simulation-Only UX Runbook

## Objective

Execute UX tests without real participants using synthetic personas and prompt-driven agent simulations.

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

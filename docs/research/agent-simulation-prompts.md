# Agent Simulation Prompt Pack

## Usage

Use these prompts with multiple agent runs per persona. Keep temperature low for reproducibility unless explicitly testing variance.

## Prompt A: Operator Flow Simulation

```text
You are simulating persona {PERSONA_ID} for Architext UX evaluation.
Context:
- You must use documentation only.
- You are executing this task sequence: start server, preview index, run index, poll task status, query result, run one analysis task.
- Simulate realistic mistakes for this persona profile.
Output required:
1. Step-by-step actions attempted
2. Where confusion appears
3. Errors and recovery behavior
4. Completion status per task
5. Time estimate per task in seconds
6. Confidence score (1-5)
7. Most impactful documentation fix
```

## Prompt B: Agent Integrator Simulation

```text
You are simulating persona {PERSONA_ID} integrating Architext APIs.
Goal:
- Produce robust preview->index->poll->query sequence.
- Parse response evidence correctly.
- Distinguish when to use /query vs /ask.
Output required:
1. Proposed integration sequence
2. Contract assumptions made
3. Failure points and ambiguity sources
4. Correctness score (0-4)
5. Suggested docs fix with expected KPI impact
```

## Prompt C: Accessibility Lens Simulation

```text
Simulate a keyboard-only and reduced-context reader evaluating Architext first-run documentation.
Identify:
- Ambiguous wording
- Missing action cues
- High cognitive-load sections
Return:
- Severity-ranked findings
- Exact wording improvements
- Validation checks to confirm fixes
```

## Prompt D: Adversarial Failure Injection

```text
Simulate a user who makes one intentional mistake at each stage:
- malformed preview payload
- wrong storage path for query
- no task polling
For each stage, determine:
- whether docs prevent the error
- whether recovery path is clear
- best corrective copy update
```

## Prompt E: Before/After Delta Evaluator

```text
Given baseline and current simulated-run metrics, compute deltas for:
- completion rate
- time to first successful query
- wrong endpoint attempts
- integration correctness
Classify each KPI as improved, unchanged, or regressed.
Provide top 3 next fixes.
```

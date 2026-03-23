# Architext UX Evaluation and Improvement Plan

## 1. Purpose and Scope

This plan adapts the UX researcher template into a practical workflow for Architext.

Primary outcome:
- Improve first-run success, task discoverability, and confidence for two user groups: human operators and agent developers.

In-scope surfaces:
- README quickstart and setup flow
- API docs and Swagger usage (`/docs`)
- Core endpoints (`/index/preview`, `/index`, `/query`, `/ask`, `/tasks/{id}`, `/providers`, `/mcp/tools`)
- Task output usability (especially `start_here`, health, security, and structure outputs)

Out of scope (for this cycle):
- Core retrieval/ranking model changes not tied to UX issues
- New large features unrelated to user friction findings

## 2. Research Questions

1. Can a new user complete the first value loop without assistance?
2. Do users understand when to use `/index/preview` before `/index`?
3. Can users find and execute the right analysis task for common goals?
4. Do agent developers understand output schemas quickly enough to integrate safely?
5. Where do users fail due to wording, information architecture, defaults, or missing feedback?

## 3. Success Metrics

Quantitative targets for this cycle:
- First-run success rate: >= 85% (from clean setup to first useful answer)
- Time to first successful query: <= 15 minutes median
- Wrong-endpoint attempts per session: <= 1 median
- Task selection accuracy for scenario prompts: >= 80%
- Documentation findability (correct section reached in <= 3 steps): >= 85%

Qualitative targets:
- Fewer reports of confusion around index lifecycle and storage paths
- Higher self-reported confidence in choosing tasks and reading outputs
- Clearer understanding of how operator flow differs from agent integration flow

## 4. Personas and Simulation Matrix

Primary persona group A: Operator (synthetic)
- Profile: engineer or technical PM using API and docs to analyze a codebase
- Simulation set: OP-N1, OP-I1, OP-A1
- Run target: 12 runs (4 per persona)

Primary persona group B: Agent Integrator (synthetic)
- Profile: developer wiring Architext outputs into another agent/tool
- Simulation set: AG-N1, AG-I1, AG-A1
- Run target: 12 runs (4 per persona)

Accessibility requirement:
- Include dedicated accessibility-lens simulations (keyboard-only and reduced-context reading)

Total run target per cycle:
- 24 core persona runs + 8 accessibility/adversarial runs

## 5. Test Sequence (Execution Order)

Run the sequence in order. Do not skip earlier steps, because each later test depends on baseline observations.

### Test 0: Baseline Instrumentation and Rubric Setup (Day 1)

Goal:
- Make sure every subsequent session produces comparable evidence.

Actions:
- Define logging sheet for completion, time, errors, confusion points, confidence score (1-5)
- Define severity rubric for issues: Critical, High, Medium, Low
- Freeze test environment (fresh clone, fresh virtual environment, default config)

Output:
- `docs/research/baseline-rubric.md` (or equivalent research notes location)

### Test 1: Heuristic Review (Expert) (Day 1-2)

Method:
- 2 evaluators run a heuristic audit across docs and API experience

Focus heuristics:
- Clarity of next action
- Consistency of terminology (`index`, `storage`, `source`, `task`)
- Visibility of system status for async tasks
- Error prevention and recovery guidance
- Accessibility and readability

Entry points:
- `README.md`
- `docs/DEVELOPMENT.md`
- Swagger UI (`/docs`)

Output:
- Prioritized heuristic issue list with evidence links and affected user group

### Test 2: First-Run Usability Test (Simulated Operator Runs) (Day 3-4)

Simulation runs:
- 12 runs across OP-N1, OP-I1, OP-A1

Scenario tasks:
1. Install and run server locally
2. Preview index plan for `./src`
3. Execute index with explicit storage path
4. Ask architecture question and identify source evidence
5. Run one analysis task and explain the result

Metrics:
- Task completion rate, time on task, error count, wrong-endpoint attempts

Pass threshold:
- At least 85% run completion for tasks 1-4

### Test 3: Agent Integration Usability Test (Simulated Integrator Runs) (Day 4-5)

Simulation runs:
- 12 runs across AG-N1, AG-I1, AG-A1

Scenario tasks:
1. Discover available tools/endpoints for integration
2. Produce a minimal integration call sequence (`preview -> index -> query`)
3. Parse one structured response and extract source references
4. Identify how to poll async task status and detect completion

Metrics:
- Integration correctness score (0-4 checklist)
- Misinterpretation count of response fields
- Time to working request sequence

Pass threshold:
- Median integration correctness >= 3/4

### Test 4: Accessibility and Inclusive UX Validation (Day 5)

Method:
- Keyboard-only walkthrough and screen-reader spot checks on docs and Swagger

Checks:
- Heading structure and landmark clarity
- Keyboard navigability for docs sections
- Readability and contrast of key content blocks
- Alternative wording clarity for non-native English readers

Output:
- Accessibility findings with direct remediation suggestions

### Test 5: Simulation Confirmation Run (Week 2)

Simulation runs:
- 16 additional runs focused on previously failing scenarios

Method:
- Prompt-based scenario reruns using fixed prompt versions
- Collect completion, time, error, and confidence outputs

Goal:
- Confirm improvements from earlier fixes generalize beyond moderated conditions

### Test 6: Regression Check After UX Changes (Week 2)

Method:
- Re-run key tasks from Test 2 and Test 3 on updated docs/flows

Goal:
- Verify no newly introduced confusion points
- Compare before/after metrics for each critical step

## 6. Improvement Backlog Framework

Translate every finding into one backlog item with:
- Problem statement
- Evidence (quote, metric, observed failure)
- Proposed change
- Expected impact
- Effort estimate (S/M/L)
- Validation test to confirm improvement

Prioritization model:
- Priority score = (Impact x Frequency x Confidence) / Effort
- Implement in descending score, but always fix Critical issues first

## 7. Prompt Set for UX Evaluation and Improvement

Use these prompts directly with an LLM assistant to keep research outputs consistent.

### Prompt 1: Research Plan Generator

```text
You are a UX researcher for Architext (API-first codebase analysis platform).
Generate a mixed-methods research plan focused on first-run usability and task discoverability.
Include:
- Research questions
- Participant criteria for Operator and Agent Integrator personas
- Session scripts
- Quantitative and qualitative metrics
- Bias and risk controls
- Accessibility inclusion criteria
Constrain recommendations to what can be validated in 2 weeks.
```

### Prompt 2: Heuristic Audit Prompt

```text
Evaluate Architext's user experience using Nielsen-style heuristics and API DX heuristics.
Input surfaces: README quickstart, docs/DEVELOPMENT.md, Swagger flow for /index/preview, /index, /query, /tasks/{id}.
Return:
- Top 10 issues ordered by severity
- For each issue: evidence, affected persona, likely root cause, suggested fix
- A "fix first" shortlist for this week
```

### Prompt 3: Moderated Session Script Prompt

```text
Create a 60-minute moderated usability testing script for Architext Operator persona.
Structure:
- Intro and consent (5m)
- Baseline questions (10m)
- Task scenarios (35m)
- Debrief (10m)
Each task must include success criteria, observation focus, and measurable metrics.
```

### Prompt 4: Agent Integrator Script Prompt

```text
Create a usability test protocol for developers integrating Architext into other agents.
Focus on:
- Endpoint discovery and call sequencing
- Parsing structured responses and source attribution
- Async task status polling reliability
Provide a scoring rubric (0-4) per participant and common failure signatures.
```

### Prompt 5: Accessibility Review Prompt

```text
Review Architext documentation and API interaction flow for accessibility and inclusive language.
Check keyboard-only navigation, heading hierarchy, readability, cognitive load, and terminology clarity.
Return issues with severity, impacted users, and concrete wording/layout fixes.
```

### Prompt 6: Finding-to-Backlog Translator

```text
Convert these UX findings into an implementation backlog.
For each finding provide:
- Problem
- Evidence
- Proposed change
- Effort (S/M/L)
- Expected KPI movement
- Verification test
Order backlog by (Impact x Frequency x Confidence) / Effort.
```

### Prompt 7: Before/After Evaluation Prompt

```text
Given baseline and post-change metrics for Architext UX tests, compute improvement deltas.
Highlight:
- Which metrics met target
- Which failed target
- Which changes likely caused improvements
- Remaining top 5 UX risks
Output an executive summary and a technical appendix.
```

### Prompt 8: Release Readiness UX Gate

```text
Act as a UX gate reviewer for an Architext release.
Decide GO / CONDITIONAL GO / NO-GO based on:
- First-run success rate
- Time to first successful query
- Task discoverability accuracy
- Accessibility issue severity
If not GO, list mandatory fixes and the minimum retest set.
```

## 8. Evidence Capture Template

Use this structure for each test run:

```markdown
# UX Test Run - [Date]

## Context
- Persona:
- Test ID:
- Environment:
- Facilitator:

## Task Outcomes
1. Task:
   - Completed: Yes/No
   - Time:
   - Errors:
   - Notes:

## Observations
- Behavioral notes:
- Confusion points:
- Quotes:

## Metrics
- Confidence (1-5):
- Satisfaction (1-5):

## Candidate Fixes
- Fix idea:
- Expected impact:
- Verification plan:
```

## 9. Implementation Cadence

Week 1:
- Test 0-4
- Produce prioritized backlog
- Implement top Critical and High UX fixes

Week 2:
- Test 5-6
- Measure before/after impact
- Finalize release UX gate decision

Simulation-only operating note:
- Execution uses synthetic personas and agent prompt runs rather than live participant interviews.
- See `docs/research/simulation-runbook.md` and `docs/research/agent-simulation-prompts.md`.

## 10. Deliverables

At the end of this cycle, produce:
- UX Findings Report
- Prioritized UX Backlog
- Implemented UX changes with references
- Before/after metric comparison
- Next-cycle research recommendations

## 11. Next Phase: Continuous Monitoring

After completing planned cycles and simulation gate approval, move to release-based monitoring:

- Run a lightweight simulation cycle each release.
- Use KPI threshold checks to trigger escalation.
- Escalate to full-cycle simulation when regression thresholds are crossed.
- Keep backlog and documentation updates in the same change cycle.

Reference:
- `docs/research/phase-2-continuous-improvement.md`

## 12. Current Status and Remaining Points

Current status:
- Phase-1 simulation campaign completed (cycles 1-5) with final GO decision.
- Core onboarding/documentation improvements implemented and validated.
- Phase-2 monitoring model and gate automation are in place.

Completed points:
- Synthetic persona matrix and prompt pack operationalized.
- KPI summarizer implemented with multi-cycle parsing and threshold checks.
- One-command UX release gate workflow implemented.
- Consolidated findings created and stale one-off filled artifacts retired.

Remaining points (phase-2 execution backlog):
1. Run release-gate check for each release candidate and append a gate log entry.
2. Recalibrate thresholds after three additional release cycles if KPI drift appears.
3. Escalate to full-cycle simulation automatically on any threshold failure.
4. Keep generated exports out of commits unless explicitly required for release artifacts.
5. Quarterly cleanup of research workspace to keep only canonical templates and summaries.

Immediate next actions:
1. Use `scripts/run_ux_release_gate.py` during each release candidate review.
2. Record gate outcomes in `docs/research/release-gate-log.md`.
3. Open/close backlog items only from gate findings and simulation evidence.

## 13. Notes on Template Alignment

This plan mirrors the source UX template by preserving:
- Methodology-first structure
- Explicit participant criteria
- 60-minute moderated protocol style
- Mixed qualitative and quantitative evidence
- Structured recommendation and success-metric outputs

Adaptation for current cycle:
- Participant criteria are represented as synthetic persona definitions and run matrices.
- Moderated session logic is preserved as scenario scripts executed by agents.

It is adapted to Architext by centering on API-first workflows, async task patterns, and operator/agent integration needs.

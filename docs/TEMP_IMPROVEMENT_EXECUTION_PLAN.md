# Temporary Improvement Execution Plan

Date: 2026-03-23
Scope: Consolidated improvements from retrospective findings and comparative analysis.

## Objective

Provide a short-cycle execution plan for reliability, analysis depth, and developer onboarding improvements while preserving current release stability.

Comparative-derived goals:

1. Keep semantic depth advantage while reducing operational cost.
2. Improve onboarding ergonomics without abandoning server-first architecture.
3. Define measurable performance/resource targets and track trend changes by release.

## Priority Queue

### Batch A (Reliability and Correctness)

- Expand AST-first coverage for high-value heuristic checks.
- Add/extend regression tests for edge cases and deterministic outputs.
- Preserve current API response contracts.

### Batch B (Performance and Scale)

- Add selective-indexing strategy for large repositories.
- Improve cache invalidation and partial refresh behavior.
- Define benchmark scenarios (small/medium/large repos).

### Batch C (Onboarding and DX)

- Improve "start-here" guidance in analysis output.
- Add clear entry-point recommendations in task outputs.
- Improve operator-facing docs for common workflows.

### Batch D (Incremental Indexing Exploration)

- Prototype file-watcher/event-driven incremental indexing.
- Compare full re-index vs incremental update timings.
- Document operational trade-offs and fallback behavior.

### Batch E (Performance and Resource Benchmarking)

- Establish repeatable benchmark harness for small/medium repository profiles.
- Capture indexing latency distribution and query latency distribution.
- Capture peak memory and CPU usage for indexing/query paths.
- Publish benchmark outputs in release documentation.

### Batch F (Constrained-Mode Feasibility)

- Design and prototype a constrained analysis mode for low-resource environments.
- Define feature parity boundaries versus full semantic mode.
- Validate that constrained mode degrades gracefully and preserves output contracts.

### Batch G (Positioning and Integration Fit)

- Clarify architecture guardrails (API-first, agent-native, server-first).
- Improve "start-here" output strategy to reduce onboarding friction.
- Define integration patterns with complementary onboarding workflows/tools.

## Quality Gates (Per Batch)

1. `python -m pytest -q`
2. `python -m ruff check .`
3. `python -m mypy src`
4. Update docs for behavior/API/config changes.

## Prompt Templates

### Prompt Template 1: Batch Implementation

You are the implementation lead for Architext.

Task Batch: [A/B/C/D]
Scope: [files/modules]
Constraints:
- Preserve public API behavior unless explicitly approved.
- Keep changes atomic and reviewable.
- Add regression tests for each behavioral change.
- Update docs in the same change when behavior/config/UX is affected.

Steps:
1. List exact files to change.
2. Implement the smallest complete set of changes.
3. Run targeted tests first.
4. Run full gates (`pytest`, `ruff`, `mypy`).
5. Produce a short change packet (what changed, why, risk, rollback).

Output format:
- Scope
- Changes
- Validation evidence
- Risks
- Rollback plan

### Prompt Template 2: Quality Review

You are the quality reviewer for Architext.

Review the latest batch for:
1. Correctness regressions
2. Contract/schema changes
3. Security impact
4. Performance impact
5. Documentation completeness

Decision rule:
- Default to NEEDS WORK unless evidence supports PASS.

Output format:
- Findings by severity (Critical/High/Medium/Low)
- Required fixes
- PASS/NEEDS WORK verdict

### Prompt Template 3: Release Readiness Check

You are the release gatekeeper.

Validate:
1. Full quality gates passed.
2. No unresolved critical/high issues.
3. Changelog and release notes updated.
4. README/DEVELOPMENT/PROJECT_STATUS consistency.

Output format:
- Checklist
- Evidence
- GO/NO-GO decision
- Follow-up actions

### Prompt Template 4: Benchmark and Resource Profile

You are the performance lead for Architext.

Task Batch: [E]
Inputs:
- Repository set: [small, medium, stress]
- Environment profile: [CPU/RAM, local/cloud model]

Deliverables:
1. Indexing latency summary (p50/p95)
2. Query latency summary (p50/p95)
3. Peak memory and CPU profile
4. Regressions/improvements vs previous baseline

Constraints:
- Use repeatable command sequence.
- Keep data collection scripts deterministic.
- Do not alter production behavior during measurement.

Output format:
- Benchmark matrix
- Resource profile
- Bottleneck analysis
- Optimization proposals

### Prompt Template 5: Architecture Guardrail Review

You are the architecture reviewer for Architext.

Task Batch: [G]

Validate:
1. Server-first/API-first boundaries are preserved.
2. Agent-native output contracts are unchanged.
3. Onboarding improvements do not force a CLI-first pivot.
4. Comparative differentiation is still clear (semantic platform focus).

Output format:
- Guardrail checklist
- Violations found
- Required changes
- PASS/NEEDS WORK verdict

### Prompt Template 6: Constrained-Mode Design Review

You are the reliability and DX reviewer.

Task Batch: [F]

Validate:
1. Constrained mode has clear scope and fallback contract.
2. Output schema compatibility with core mode is preserved.
3. Risks (false positives/negatives, quality degradation) are explicitly measured.

Output format:
- Mode definition
- Compatibility matrix
- Risk assessment
- Recommended rollout strategy

## Tracking Table

| Date | Batch | Change Summary | Tests | Ruff | Mypy | Verdict | Notes |
|------|-------|----------------|-------|------|------|---------|-------|
| 2026-03-23 | E | Added repeatable benchmark harness (`scripts/benchmark.py`) and generated matrix artifacts (`docs/benchmarks/*`) | PASS (309) | PASS | PASS | Continue | small index p50/p95: 4.5209/5.835s; medium index p50/p95: 35.1885/36.141s; query p50/p95: small 0.1333/0.1373s, medium 0.1307/0.1364s |
| 2026-03-23 | A | Expanded AST-first security heuristics and taint-flow reliability checks with regression coverage | PASS (312) | PASS | PASS | Continue | Added detections for `subprocess(..., shell=True)` and unsafe `yaml.load(...)`; taint detection now includes keyword args and f-string sources |
| 2026-03-23 | B | Added selective indexing limits/extension filters and source-aware cache invalidation behavior | PASS (315) | PASS | PASS | Continue | New config knobs: `INDEX_MAX_FILES`, `INDEX_INCLUDE_EXTENSIONS`; cache invalidation now correctly scopes by `source_path` |
| 2026-03-23 | C | Added structure-analysis onboarding hints and documented operator start-here workflow | PASS (315) | PASS | PASS | Continue | `analyze-structure` now returns `start_here` recommendations for practical entry points |
| 2026-03-23 | F | Prototyped constrained analysis mode with contract-compatible structure outputs | PASS (316) | PASS | PASS | Continue | Added `analysis_mode` and `constrained_max_files`; constrained runs preserve `format/summary/tree/start_here` |

## Suggested Batch Order

1. A -> correctness and safety
2. B -> scale/performance internals
3. E -> benchmark and resource baseline publication
4. C -> onboarding and "start-here" output improvements
5. F -> constrained-mode feasibility
6. D -> incremental indexing prototype
7. G -> architecture/positioning guardrail verification

# Heuristic Review - 2026-03-23

## Scope

- `README.md`
- `docs/DEVELOPMENT.md`
- Swagger `/docs` request flow expectations

## Method

- Expert heuristic pass based on clarity, consistency, status visibility, error prevention, and accessibility/readability.

## Findings

### UX-H1 (High)

- Problem: Async index flow lacked an explicit polling example in quickstart context.
- Evidence: Users can execute `/index` but still miss `GET /tasks/{id}` follow-up behavior in README.
- Affected persona: Operator, Agent Integrator
- Recommendation: Add a concrete task-status polling example tied to `/index` output.
- Status: Done (implemented and verified in internal pilot)

### UX-H2 (High)

- Problem: First-run flow steps were present but distributed across sections.
- Evidence: Setup and endpoint usage were separated by endpoint groups; no single short flow block.
- Affected persona: Operator
- Recommendation: Add a compact first-value loop sequence.
- Status: Done (implemented in current cycle)

### UX-H3 (Medium)

- Problem: Common first-run failures were not grouped in one place.
- Evidence: Troubleshooting hints were embedded but not consolidated.
- Affected persona: Operator
- Recommendation: Add first-run troubleshooting matrix with symptom-cause-fix mapping.
- Status: Done (implemented in current cycle)

### UX-H4 (Medium)

- Problem: UX plan execution assets were not discoverable from canonical developer docs.
- Evidence: Temporary plan was referenced, but operational assets folder did not exist.
- Affected persona: Internal maintainers
- Recommendation: Add `docs/research/` asset pack and link from `docs/DEVELOPMENT.md`.
- Status: Done (implemented in current cycle)

### UX-H5 (Low)

- Problem: API endpoint decision point between `/query` and `/ask` may remain ambiguous for new users.
- Evidence: Both are presented but trade-offs are not summarized in README.
- Affected persona: Operator, Agent Integrator
- Recommendation: Add a one-line decision hint for `/query` vs `/ask`.
- Status: Done (implemented in current cycle)

## Next Actions

1. Run simulation batches using `docs/research/synthetic-personas.md` and `docs/research/agent-simulation-prompts.md`.
2. Compare simulation findings with pilot notes in `docs/research/internal-pilot-test2-test3-2026-03-23.md`.
3. Re-score backlog items based on observed simulation frequency and rerun targeted scenarios.

# Agent Integrator Moderated Session Script (60 Minutes)

## 1. Introduction and Consent (5m)

Moderator prompts:
- "We are evaluating integration usability, not your performance."
- "Please narrate your reasoning while you work."

## 2. Baseline Questions (10m)

- What is your typical pattern for integrating API tools into agents?
- How do you usually validate response contracts?
- What failure modes are most costly in your integrations?

## 3. Task Scenarios (35m)

### Task 1: Discover Integration Surface

Prompt:
- "Identify which endpoints and outputs are required for a minimal integration."

Success criteria:
- Participant identifies `preview -> index -> query` and async status checks

Scoring (0-4):
- 1 point for each correctly identified step:
  - preview
  - index
  - task polling
  - query/ask

### Task 2: Produce Call Sequence

Prompt:
- "Write a minimal request sequence for a robust first integration."

Success criteria:
- Sequence includes payload basics and error-aware ordering

Observation focus:
- Schema comprehension
- Handling of storage/source consistency

### Task 3: Parse Structured Response

Prompt:
- "Parse a response and extract answer plus source references."

Success criteria:
- Accurate extraction with correct field interpretation

Observation focus:
- Misread field names
- Confidence in source attribution

### Task 4: Async Reliability Handling

Prompt:
- "Show how your integration detects completion/failure for async tasks."

Success criteria:
- Reliable polling strategy and failure branch handling

Observation focus:
- Poll interval assumptions
- Terminal-state detection correctness

## 4. Debrief (10m)

- Which integration step was least clear?
- Which schema fields were ambiguous?
- What docs change would improve correctness most?

## 5. Failure Signatures

Track these common signatures:
- Skipped preview step
- Query against wrong/nonexistent storage
- No async polling or incorrect terminal state handling
- Mis-parsing source evidence fields

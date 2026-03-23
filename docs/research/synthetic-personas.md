# Synthetic Personas for Agent-Based UX Simulation

## Purpose

Define reproducible synthetic personas for simulation-only UX execution.

## Operator Personas

### OP-N1: Novice Operator
- API experience: low
- Mental model: endpoint-by-endpoint, minimal async intuition
- Risk profile: payload formatting errors, misses polling step
- Accessibility profile: keyboard-only navigation

### OP-I1: Intermediate Operator
- API experience: medium
- Mental model: understands request/response but inconsistent with storage path reuse
- Risk profile: query against wrong storage, mixed source paths

### OP-A1: Advanced Operator
- API experience: high
- Mental model: reliable task flow, expects schema clarity
- Risk profile: edge-case ambiguity and fallback handling

## Agent Integrator Personas

### AG-N1: Novice Integrator
- Integration experience: low
- Mental model: synchronous assumptions
- Risk profile: skips async lifecycle handling and terminal state checks

### AG-I1: Intermediate Integrator
- Integration experience: medium
- Mental model: can chain endpoints, may misread compact mode differences
- Risk profile: `/query` vs `/ask` field interpretation drift

### AG-A1: Advanced Integrator
- Integration experience: high
- Mental model: contract-first implementation
- Risk profile: only fails on ambiguous docs and missing examples

## Scenario Variation Axes

Use these axes to generate run diversity:
- Source size: small, medium
- Storage mode: default, explicit
- Request style: Swagger examples, handcrafted payload
- Failure seed: malformed payload, missing storage, premature query
- Accessibility lens: keyboard-only comprehension and reduced-context reading

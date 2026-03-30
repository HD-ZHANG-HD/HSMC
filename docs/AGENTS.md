# AGENTS.md

## Purpose

This repository-level handoff file is the first entrypoint for a new Codex session.

It consolidates the repository-level guidance and docs-entrypoint notes that were previously spread across multiple markdown files.

Documentation under `docs/` contains project documentation and persistent design information.
No executable code should be placed there.

## Project Handoff Quickstart

Primary working area:

- `he_compiler/operator_execution_framework`

Read these files first:

1. `AGENTS.md`
2. `docs/handoff_status.md`
3. `docs/backend_capability_matrix.md`
4. `docs/architecture.md`

## What Each File Is For

- `docs/architecture.md`
  - system architecture overview
  - operator design model
  - restricted HE/NEXUS contracts
  - compiler min-cut prototype reference
- `docs/backend_capability_matrix.md`
  - operator/backend status
  - real/restricted binding details
  - attention MPC integration summary
- `docs/handoff_status.md`
  - concise current project state for new agent sessions
  - where important code lives
  - build and validation anchors
  - current limitations and next milestone

## Rules for Next Session

1. Keep changes incremental: integrate one operator at a time.
2. Preserve router contract:
   - `OperatorRegistry -> backend -> operator_fn(inputs, ctx)`
3. Do not move routing or domain conversion logic into operators.
4. Update capability registry whenever backend status changes:
   - `runtime/capabilities.py`
5. Update `docs/backend_capability_matrix.md` and `docs/handoff_status.md` whenever executable status changes.
6. For each real operator integration, always run staged validation:
   - standalone bridge test
   - python wrapper test
   - routed framework test
7. Do not claim real HE<->MPC conversion unless cryptographic domain conversion is truly implemented.
8. If a path is restricted or approximate, label it honestly rather than broadening claims.

## Current Real MPC Operators

- `GeLU`
- `Softmax`
- `LayerNorm`
- `FFN_Linear_1`
- `FFN_Linear_2`
- `Attention_QK_MatMul`
- `Attention_V_MatMul`
- `Residual_Add`

## Current Restricted or Approximate HE / NEXUS Operators

- `GeLU` (approximate)
- `Softmax` (approximate)
- `LayerNorm` (restricted)
- `FFN_Linear_1` (restricted)
- `FFN_Linear_2` (restricted)
- `Attention_QK_MatMul` (restricted)
- `Attention_V_MatMul` (restricted)
- `Residual_Add` (semantic add over backend-native add)

## Current Mock Operators

- `Embedding`
- `Linear_QKV`
- `Out_Projection`

## Current Best Next Task

Integrate `Out_Projection@MPC` as the next single real linear operator, following the same bridge+wrapper+routed-test workflow used for `FFN_Linear_1`.

After that, the main remaining honest gaps are:

- `Linear_QKV`
- `Out_Projection`
- `Embedding` only if a real backend-specific path can be supported honestly

## Build/Test Anchors

Build bridges from:

- `he_compiler/EzPC_bolt/EzPC/SCI/build`

Run validations from:

- `he_compiler/operator_execution_framework`

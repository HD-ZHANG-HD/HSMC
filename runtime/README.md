# runtime/

## Purpose

Implements the execution runtime.

## Responsibilities

- Operator routing.
- Backend selection.
- Execution context management.
- Domain conversion (HE <-> MPC).

## Design Notes

Runtime code remains backend-agnostic and should route by operator-level plans.
It should support future method-level dispatch, where each operator may expose multiple methods and the runtime dispatches the selected method for execution.

# operators/

## Purpose

Contains operator definitions and their implementations.

Operators are the primary abstraction of the framework. Each operator has its own folder (for example `softmax/`, `gelu/`, `layernorm/`).

## Operator Directory Contract

Inside each operator directory:

- `spec.py`: defines operator metadata, including inputs, outputs, and attributes.
- Method implementation modules: execution variants such as `method_mpc_bolt.py`, `method_mpc_alt.py`, or `method_he_poly.py`.
- Optional bridge wrappers when needed by specific methods.

## Method-Level Dispatch Readiness

This layout is designed to support multiple execution methods per operator, so planning/compilation can later choose methods at operator granularity.

# FFN_Linear_1-Only Real Integration Notes

## Why FFN_Linear_1 was selected first

Between `FFN_Linear_1` and `Out_Projection`, `FFN_Linear_1` is the easier first linear target because it is a pure affine operator (no residual coupling) and maps directly to a standalone MPC matrix multiply primitive.

## Exact wrapped source and function

- Source file: `he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.cpp`
- Header: `he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.h`
- Constructor used: `NonLinear::NonLinear(int party, string address, int port)`
- Wrapped core primitive:
  - `NonLinear::n_matrix_mul_iron(...)`

Bridge applies:
1. real MPC matrix multiply via `n_matrix_mul_iron`,
2. bias addition in shared ring.

## Input/output shape mapping

Framework operator input shape is assumed `[B, S, H]`.

- Number of independent matmuls: `n = B * S`
- Each matmul is `1 x H` times `H x I` -> `1 x I`
- Bridge params:
  - `n` = `B*S`
  - `h` = `H`
  - `i` = `I` (output dimension)
- Output reshaped back to `[B, S, I]`.

## Weight handling

Weights/bias are generated deterministically in wrapper (no model file needed):

- `W ~ N(0,1)` shape `[H, I]`
- `b ~ N(0,1)` shape `[I]`
- fixed seed (default `1234`, configurable)

For `n_matrix_mul_iron`, `W` and `b` are replicated across the `n` independent row-matmuls.

## Auxiliary buffers

- No auxiliary output buffers are required.
- Only output share buffer is produced/recombined.

## Router contract (unchanged)

- `OperatorRegistry -> backend -> operator_fn(inputs, ctx)` remains unchanged.
- Only `FFN_Linear_1@MPC` backend dispatch was switched to real wrapper.

## Capability/status registry

Added explicit queryable registry:

- file: `framework/capabilities.py`
- statuses per `(operator, backend)`:
  - `real-integrated`
  - `mock`
  - `unsupported`

`FFN_Linear_1@MPC` is marked `real-integrated`.

## What remains mocked

- Other linear operators (`Out_Projection`, `Linear_QKV`, attention matmuls) remain mock in this step.
- HE/MPC conversion remains orchestration-level only (not real domain conversion runtime).

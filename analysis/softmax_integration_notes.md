# Softmax-Only Real Integration Notes

## Exact wrapped source and function

- Source file: `he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.cpp`
- Header: `he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.h`
- Constructor used: `NonLinear::NonLinear(int party, string address, int port)`
- Wrapped function:
  - `NonLinear::softmax(int nthreads, uint64_t* input, uint64_t* output, uint64_t* l, int dim, int array_size, int ell, int s)`

## Meaning of additional `l` buffer

`l` is the auxiliary output produced by `fpmath->softmax_fix(...)` (returned as `l_short` internally in `nonlinear.cpp`).

- In bert_bolt full pipeline, this buffer is used in downstream pruning/debug logic.
- In this framework step, `Softmax` operator returns a single tensor, so:
  - `l` is computed for real,
  - `l` is recombined and validated inside the wrapper,
  - `l` is **kept internal** (not exposed from operator output).

## Shape assumptions and row mapping

Softmax is applied row-wise on last axis:

- framework input shape: `[..., K]`
- wrapper maps to bert_bolt expected 2D view:
  - `dim = product(all axes except last)`
  - `array_size = K`
  - input reshaped to `[dim, array_size]` (row-major)
- bert_bolt computes one softmax per row of length `array_size`.
- output reshaped back to original framework shape.

This preserves softmax semantics and avoids incorrect full flattening.

## Buffer conversion details

1. Input float tensor -> fixed-point with scale `s`
2. Ring encode under `ell` bits into `uint64_t`
3. Create 2-party additive shares
4. Launch two bridge processes (party 1/2)
5. Run real `NonLinear::softmax(...)` on both shares
6. Recombine output shares (`output`)
7. Recombine auxiliary shares (`l`) internally for validation
8. Decode signed fixed-point values and reshape

## Router invocation path (unchanged contract)

- `OperatorRouter.execute_pipeline(...)`
- `OperatorRegistry.get("Softmax", BackendType.MPC)`
- backend function `_mk_softmax(...).fn(inputs, ctx)`
- MPC branch calls `run_bert_bolt_softmax_mpc(...)`
- returns `TensorValue` as usual

`OperatorRegistry -> backend -> operator_fn(inputs, ctx)` is unchanged.

## Runtime parameters used

- `ell = 37`
- `s = 12`
- `nthreads = 2`
- `address = 127.0.0.1`
- `port = dynamic contiguous free block` (required because SCI uses `port + i` internally)

## Scope statement

- `Softmax@MPC` is real-integrated in this step.
- Other operators are unchanged in this step.
- No claim is made for real HE/MPC conversion.

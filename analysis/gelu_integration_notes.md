# GeLU-Only Real Integration Notes

## Exact wrapped source and function

- Source file: `he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.cpp`
- Header: `he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.h`
- Constructor used: `NonLinear::NonLinear(int party, string address, int port)`
- Wrapped function: `NonLinear::gelu(int nthreads, uint64_t* input, uint64_t* output, int size, int ell, int s)`

## Runtime setup requirement (SCI)

`NonLinear` internally creates `MAX_THREADS` IO/OT packs (`port + i` for `i in [0,63]`), so the wrapper must:

1. launch both parties (`party=1`, `party=2`),
2. use the same base `address` and `port`,
3. ensure a free contiguous port block is available.

No additional explicit initialization call is required beyond constructing `NonLinear`.

## Tensor shape assumptions

- GeLU input tensor shape is assumed to be `[B, S, H]`.
- Flattened MPC size is:
  - `size = B * S * H`.
- Output is reshaped back to `[B, S, H]`.

## Buffer conversion details

1. Input float tensor -> fixed-point:
   - `q = round(x * 2^s)`.
2. Ring encoding:
   - `q_u64 = q & ((1 << ell) - 1)`.
3. Two-party additive secret shares:
   - sample random `share0`,
   - `share1 = (q_u64 - share0) mod 2^ell`.
4. Each share is written as a raw `uint64_t` binary buffer.
5. Bridge process calls `NonLinear::gelu(...)` on each party share.
6. Recombine output shares:
   - `y_u64 = (y0 + y1) mod 2^ell`.
7. Signed decode and dequantize:
   - convert from two's complement under `ell`,
   - divide by `2^s`.

## Router invocation path (unchanged contract)

- `OperatorRouter.execute_pipeline(...)`
- `OperatorRegistry.get("GeLU", BackendType.MPC)`
- backend function `_mk_gelu(...).fn(inputs, ctx)`
- for MPC only, backend calls:
  - `run_bert_bolt_gelu_mpc(...)` in `framework/gelu_real_mpc.py`
- returns `TensorValue` normally.

The `OperatorRegistry -> backend -> operator_fn(inputs, ctx)` interface is unchanged.

## What remains mocked

- All operators except `GeLU@MPC` remain NumPy/mock implementations.
- HE/MPC conversion in router is still orchestration-level metadata conversion, not SCI/HE runtime conversion.
- No claim of real HE/MPC conversion is made in this step.

## Chosen runtime values in validation

- `ell = 37`
- `s = 12`
- `nthreads = 2`
- `address = 127.0.0.1`
- `port = dynamic free block` in wrapper (or explicit fixed value in standalone bridge test)

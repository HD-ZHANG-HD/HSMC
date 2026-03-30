# LayerNorm-Only Real Integration Notes

## Exact wrapped source and function

- Source file: `he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.cpp`
- Header: `he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/nonlinear.h`
- Constructor used: `NonLinear::NonLinear(int party, string address, int port)`
- Wrapped function:
  - `NonLinear::layer_norm(int nthreads, uint64_t* input, uint64_t* output, uint64_t* weight, uint64_t* bias, int dim, int array_size, int ell, int s)`

## Tensor shape assumptions and mapping

LayerNorm is applied on the last axis:

- framework input shape: `[..., K]`
- mapped to bert_bolt row format:
  - `dim = product(all axes except last)`
  - `array_size = K`
  - row-wise reshape to `[dim, array_size]`
- output reshaped back to original framework shape.

## Affine parameter handling

bert_bolt LayerNorm requires affine buffers (`weight`, `bias`) with total size `dim * array_size`.

Wrapper supports:

- `layernorm_weight` / `layernorm_bias` from `ExecutionContext.params` as:
  - shape `[K]` (broadcast to each row), or
  - shape `[dim, K]` (full per-row affine).
- if omitted:
  - default affine is used (`weight=1`, `bias=0`) and passed as real buffers.

## Auxiliary buffers

- No additional auxiliary output buffers are required for LayerNorm bridge.
- Only output share buffer is returned/recombined.

## Buffer conversion details

1. Input / weight / bias float tensors -> fixed-point with scale `s`
2. Ring encode to `uint64_t` under bitwidth `ell`
3. Two-party additive shares generated for each buffer
4. Two bridge processes invoke real `NonLinear::layer_norm(...)`
5. Output shares are recombined and decoded back to float.

## Router invocation path (unchanged contract)

- `OperatorRouter.execute_pipeline(...)`
- `OperatorRegistry.get("LayerNorm", BackendType.MPC)`
- backend function `_mk_layer_norm(...).fn(inputs, ctx)`
- MPC branch calls `run_bert_bolt_layernorm_mpc(...)`
- returns `TensorValue`.

`OperatorRegistry -> backend -> operator_fn(inputs, ctx)` is unchanged.

## Runtime values in validation

- `ell = 37`
- `s = 12`
- `nthreads = 2`
- `address = 127.0.0.1`
- `port = dynamic contiguous free block` (SCI uses `port + i` internally).

## Scope statement

- `LayerNorm@MPC` is real-integrated in this step.
- Other operators were not changed by this step.
- No claim is made for real HE/MPC domain conversion in this step.

# Backend Capability Matrix

## Purpose

This file is the method-level capability and contract source of truth for operator execution status across `MPC`, `HE`, and `HYBRID`.

It also consolidates the attention MPC integration summary that was previously kept separately.

## Status Values

- `real-integrated`: routed to a real integrated runtime/wrapper path
- `restricted-integrated`: integrated with explicit shape/layout restrictions
- `mock`: framework numpy/mock implementation
- `unsupported`: no implementation wired

## Method-Level Matrix (Current)

This table is method-level. The runtime is still backend-routed today, but methods are now first-class in the operator layout.

| Operator | Method | Backend | Status |
|---|---|---|---|
| Embedding | method_runtime_default | MPC | mock |
| Embedding | method_runtime_default | HE | mock |
| Embedding | method_runtime_default | HYBRID | mock |
| Linear_QKV | method_runtime_default | MPC | mock |
| Linear_QKV | method_runtime_default | HE | mock |
| Linear_QKV | method_runtime_default | HYBRID | mock |
| Attention_QK_MatMul | method_mpc | MPC | real-integrated |
| Attention_QK_MatMul | method_he_nexus | HE | restricted-integrated |
| Attention_QK_MatMul | method_runtime_default | HYBRID | mock |
| Softmax | method_mpc_bolt | MPC | real-integrated |
| Softmax | method_he_nexus | HE | real-integrated |
| Softmax | method_runtime_default | HYBRID | mock |
| Attention_V_MatMul | method_mpc | MPC | real-integrated |
| Attention_V_MatMul | method_he_nexus | HE | restricted-integrated |
| Attention_V_MatMul | method_runtime_default | HYBRID | mock |
| Out_Projection | method_runtime_default | MPC | mock |
| Out_Projection | method_runtime_default | HE | mock |
| Out_Projection | method_runtime_default | HYBRID | mock |
| Residual_Add | method_runtime_default | MPC | real-integrated |
| Residual_Add | method_runtime_default | HE | real-integrated |
| Residual_Add | method_runtime_default | HYBRID | mock |
| LayerNorm | method_mpc_bolt | MPC | real-integrated |
| LayerNorm | method_he_nexus | HE | restricted-integrated |
| LayerNorm | method_runtime_default | HYBRID | mock |
| FFN_Linear_1 | method_mpc_bolt | MPC | real-integrated |
| FFN_Linear_1 | method_he_nexus | HE | restricted-integrated |
| FFN_Linear_1 | method_runtime_default | HYBRID | mock |
| GeLU | method_mpc_bolt | MPC | real-integrated |
| GeLU | method_he_nexus | HE | real-integrated |
| GeLU | method_runtime_default | HYBRID | mock |
| FFN_Linear_2 | method_mpc_bolt | MPC | real-integrated |
| FFN_Linear_2 | method_he_nexus | HE | restricted-integrated |
| FFN_Linear_2 | method_runtime_default | HYBRID | mock |

## Real-Integrated and Restricted-Integrated Method Bindings

- `GeLU.method_mpc_bolt @ MPC`
  - wrapper: `operators/gelu/method_mpc_bolt.py`
  - bridge primitive: `NonLinear::gelu(...)`
- `Softmax.method_mpc_bolt @ MPC`
  - wrapper: `operators/softmax/method_mpc_bolt.py`
  - bridge primitive: `NonLinear::softmax(...)`
- `LayerNorm.method_mpc_bolt @ MPC`
  - wrapper: `operators/layernorm/method_mpc_bolt.py`
  - bridge primitive: `NonLinear::layer_norm(...)`
- `Residual_Add.method_runtime_default @ MPC`
  - wrapper: `operators/residual_add/method_runtime_default.py`
  - execution primitive: backend-native tensor add
- `FFN_Linear_1.method_mpc_bolt @ MPC`
  - wrapper: `operators/linear_ffn1/method_mpc_bolt.py`
  - bridge primitive: `NonLinear::n_matrix_mul_iron(...)`
- `FFN_Linear_2.method_mpc_bolt @ MPC`
  - wrapper: `operators/ffn_linear_2/method_mpc_bolt.py`
  - lowered primitive: `NonLinear::n_matrix_mul_iron(...)` via `FFN_Linear_1.method_mpc_bolt`
- `Attention_QK_MatMul.method_mpc @ MPC`
  - wrapper: `operators/attention_qk_matmul/method_mpc.py`
  - bridge primitive: `NonLinear::n_matrix_mul_iron(...)`
- `Attention_V_MatMul.method_mpc @ MPC`
  - wrapper: `operators/attention_v_matmul/method_mpc.py`
  - bridge primitive: `NonLinear::n_matrix_mul_iron(...)`
- `GeLU.method_he_nexus @ HE` (approximate NEXUS-based)
  - wrapper: `operators/gelu/method_he_nexus.py`
  - bridge source mapping: `NEXUS/src/gelu.cpp` + `NEXUS/data/data_generation.py`
- `Softmax.method_he_nexus @ HE` (approximate NEXUS-based)
  - wrapper: `operators/softmax/method_he_nexus.py`
  - bridge source mapping: `NEXUS/src/softmax.cpp` + `NEXUS/src/ckks_evaluator.cpp`
- `FFN_Linear_1.method_he_nexus @ HE` (restricted NEXUS-based)
  - wrapper: `operators/linear_ffn1/method_he_nexus.py`
  - adapter bridge mapping: `backends/he_nexus_linear_ffn1_adapter.py` + `NEXUS/src/matrix_mul.cpp`
  - contract: input `[B,S,768]`, output `[B,S,64]`, with `1<=B*S<=4096`
- `LayerNorm.method_he_nexus @ HE` (restricted NEXUS-based)
  - wrapper: `operators/layernorm/method_he_nexus.py`
  - adapter bridge mapping: `backends/he_nexus_layernorm_adapter.py` + `NEXUS/src/layer_norm.cpp`
  - contract: input `[B,S,768]`, output `[B,S,768]`, with `1<=B*S<=16`, no affine weight/bias
- `Residual_Add.method_runtime_default @ HE`
  - wrapper: `operators/residual_add/method_runtime_default.py`
  - execution primitive: backend-native tensor add
- `Attention_QK_MatMul.method_he_nexus @ HE` (restricted NEXUS-based)
  - wrapper: `operators/attention_qk_matmul/method_he_nexus.py`
  - adapter bridge mapping: `backends/he_nexus_attention_adapter.py` + `NEXUS/src/matrix_mul.cpp`
  - contract: input packed `[3,B,S,768]`, output `[B,12,S,S]`, with `1<=B<=8`, `1<=S<=128`
- `Attention_V_MatMul.method_he_nexus @ HE` (restricted NEXUS-based)
  - wrapper: `operators/attention_v_matmul/method_he_nexus.py`
  - adapter bridge mapping: `backends/he_nexus_attention_adapter.py` + `NEXUS/src/matrix_mul.cpp`
  - contract: inputs `[B,12,S,S]` and packed `[3,B,S,768]`, output `[B,S,768]` (or canonical `[B,12,S,64]`)
- `FFN_Linear_2.method_he_nexus @ HE` (restricted NEXUS-based)
  - wrapper: `operators/ffn_linear_2/method_he_nexus.py`
  - lowered primitive: `MMEvaluator::matrix_mul` via `backends/he_nexus_linear_ffn1_adapter.py`
  - contract: configurable linear map `[B,S,H] -> [B,S,O]`, bounded by `1<=B*S<=4096`

## Attention MatMul MPC Summary

## Scope

This note summarizes the real MPC integrations for:

- `operators/attention_qk_matmul/`
- `operators/attention_v_matmul/`

Both operators follow the existing router contract:

- `OperatorRegistry -> backend -> operator_fn(inputs, ctx)`

No router behavior changes were introduced.

## Integrated Methods

### `Attention_QK_MatMul.method_mpc @ MPC`

- Python wrapper: `operators/attention_qk_matmul/method_mpc.py`
- Bridge binary: `BOLT_QK_MATMUL_MPC_BRIDGE`
- C++ bridge source: `bridge/qk_matmul_mpc_bridge.cpp`
- MPC primitive: `NonLinear::n_matrix_mul_iron(...)`

Computes `QK^T` as batched MPC GEMM.

### `Attention_V_MatMul.method_mpc @ MPC`

- Python wrapper: `operators/attention_v_matmul/method_mpc.py`
- Bridge binary: `BOLT_ATTN_V_MATMUL_MPC_BRIDGE`
- C++ bridge source: `bridge/attn_v_matmul_mpc_bridge.cpp`
- MPC primitive: `NonLinear::n_matrix_mul_iron(...)`

Computes `Attn * V` as batched MPC GEMM.

## Input Normalization (Compatibility-First, Future-Proof)

Both wrappers support:

1. Current packed runtime path (`qkv_out` form used by `Linear_QKV`).
2. Future pre-split canonical tensors (`Q/K/V`).

Normalization target is canonical attention layout:

- `Q, K, V`: `[B, H, S, D]`

Head handling uses runtime context:

- `ctx.params["attention_num_heads"]` (fallback: `num_heads`, then `1`)

## Tensor Shape Mapping

### `Attention_QK_MatMul`

- Canonical compute: `[B,H,S,D] x [B,H,D,S] -> [B,H,S,S]`
- Batched GEMM flattening:
  - `n = B * H`
  - `dim1 = S`
  - `dim2 = D`
  - `dim3 = S`

### `Attention_V_MatMul`

- Canonical compute: `[B,H,S,S] x [B,H,S,D] -> [B,H,S,D]`
- Batched GEMM flattening:
  - `n = B * H`
  - `dim1 = S`
  - `dim2 = S`
  - `dim3 = D`

Output behavior:

- Internal canonical output is always `[B,H,S,D]`.
- Default return is router-compatible `[B,S,H*D]`.
- Optional canonical return can be enabled with:
  - `ctx.params["attention_return_canonical"] = True`

## Build and Validation

Build from `he_compiler/EzPC_bolt/EzPC/SCI/build`:

```bash
cmake ..
cmake --build . --target BOLT_QK_MATMUL_MPC_BRIDGE BOLT_ATTN_V_MATMUL_MPC_BRIDGE -j4
```

Run from `he_compiler/operator_execution_framework`:

```bash
# standalone bridge checks
python experiments/validate_qk_matmul_bridge.py
python experiments/validate_attn_v_matmul_bridge.py

# python wrapper checks
python experiments/validate_qk_matmul_wrapper.py
python experiments/validate_attn_v_matmul_wrapper.py

# routed checks
python experiments/validate_attention_matmul_routed.py
```

Required tested shapes:

- `B=1, H=1, S=2, D=4`
- `B=1, H=4, S=8, D=16`
- `B=4, H=2, S=8, D=16`

## Embedding Status Note

`Embedding@MPC` and `Embedding@HE` remain `mock`.

This file preserves the current honest status:

- there is no current real-integrated or restricted-integrated general embedding primitive wired for the framework
- embedding semantics still come from framework/runtime-default logic rather than a backend-specific executable primitive

## Source of Truth in Code

- Runtime capability registry:
  - `runtime/capabilities.py`
- Runtime operator list:
  - `runtime/operator_specs.py`

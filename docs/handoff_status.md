# Handoff Status (Ready for Next Codex Session)

## Read This First

If a new Codex session needs to resume work immediately without reading old chat history, read these files in this order:

1. `AGENTS.md`
2. `docs/handoff_status.md`
3. `docs/backend_capability_matrix.md`
4. `docs/architecture.md`

This handoff is intentionally concise first, then more detailed where needed.

## Current State

Framework is runnable after the operator-first reorganization and preserves end-to-end behavior.

Current top-level layout under `he_compiler/operator_execution_framework`:

- `docs/`
- `configs/`
- `ir/`
- `runtime/`
- `operators/`
- `backends/`
- `bridges/`
- `experiments/`
- `tests/`

Compatibility layer:

- `framework/` remains as a facade that re-exports runtime/operator modules to keep older imports working.

## Where Key Code Lives

### Runtime / routing

- `runtime/router.py`
  - pipeline execution and runtime conversion insertion
- `runtime/operator_specs.py`
  - canonical BERT operator sequence
- `runtime/operator_registry.py`
  - backend-keyed operator implementation registry
- `runtime/capabilities.py`
  - operator/backend capability status
- `runtime/conversion/`
  - runtime HE<->MPC conversion interface, registry, manager, capability registry, and direction methods

### Backend/operator execution wiring

- `framework/backends.py`
  - main registration flow that maps operator + backend to executable functions

### Operator implementations

- `operators/attention_qk_matmul/`
- `operators/attention_v_matmul/`
- `operators/softmax/`
- `operators/gelu/`
- `operators/layernorm/`
- `operators/linear_ffn1/`
- `operators/ffn_linear_2/`
- `operators/residual_add/`

### NEXUS HE adapters

- `backends/he_nexus_attention_adapter.py`
- `backends/he_nexus_linear_ffn1_adapter.py`
- `backends/he_nexus_layernorm_adapter.py`

### Legacy execution sources reused

- `he_compiler/MPCFormer/src/benchmark/models.py`
- `he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/{bert.cpp,linear.cpp,nonlinear.cpp}`
- `he_compiler/NEXUS/src/{matrix_mul.cpp,layer_norm.cpp,softmax.cpp,gelu.cpp}`

### Compiler prototype

- `compiler/min_cut/`

## Current Operator Status Summary

### Real integrated on MPC

- `GeLU.method_mpc_bolt @ MPC`
- `Softmax.method_mpc_bolt @ MPC`
- `LayerNorm.method_mpc_bolt @ MPC`
- `FFN_Linear_1.method_mpc_bolt @ MPC`
- `FFN_Linear_2.method_mpc_bolt @ MPC` (semantic lowering to the existing FFN linear primitive)
- `Attention_QK_MatMul.method_mpc @ MPC`
- `Attention_V_MatMul.method_mpc @ MPC`
- `Residual_Add.method_runtime_default @ MPC` (semantic add)

### Real or restricted integrated on HE

- `GeLU.method_he_nexus @ HE` (approximate, NEXUS-based)
- `Softmax.method_he_nexus @ HE` (approximate, NEXUS-based)
- `LayerNorm.method_he_nexus @ HE` (restricted, NEXUS-based)
- `FFN_Linear_1.method_he_nexus @ HE` (restricted, NEXUS-based)
- `FFN_Linear_2.method_he_nexus @ HE` (restricted, NEXUS-based)
- `Attention_QK_MatMul.method_he_nexus @ HE` (restricted, NEXUS-based)
- `Attention_V_MatMul.method_he_nexus @ HE` (restricted, NEXUS-based)
- `Residual_Add.method_runtime_default @ HE` (semantic add)

### Still mock

- `Embedding`
- `Linear_QKV`
- `Out_Projection`
- any `HYBRID` operator execution beyond routing/orchestration behavior

## Important Honest Constraints

- Router conversion steps are still orchestration-level transitions, not real HE<->MPC cryptographic domain conversion.
- Method-level dispatch is represented in operator structure (`operators/*/spec.py` + method modules), but runtime selection is still backend-keyed in `runtime/router.py`.
- `Embedding@MPC` and `Embedding@HE` remain `mock`; do not overclaim them as true executable embedding operators.
- `GeLU@HE` and `Softmax@HE` are integrated via approximate NEXUS-based methods.
- `LayerNorm@HE` is restricted:
  - input `[B,S,768]`, output `[B,S,768]`, `1<=B*S<=16`
  - affine weight/bias unsupported
  - semantics follow current NEXUS layer-norm execution style and are not a fully general HE LayerNorm
- `FFN_Linear_1@HE` is restricted:
  - input `[B,S,768]`, output `[B,S,64]`, `1<=B*S<=4096`
- `FFN_Linear_2@HE` is restricted:
  - lowered to the existing NEXUS matrix-mul adapter used by `FFN_Linear_1@HE`
- `Attention_QK_MatMul@HE` and `Attention_V_MatMul@HE` are restricted:
  - packed hidden/head contract only
- `Residual_Add@MPC` and `Residual_Add@HE` are semantic add operators:
  - exact shape-preserving elementwise add
  - current execution is lowered to backend-native tensor add
- `FFN_Linear_2@MPC` is integrated by lowering to the existing `NonLinear::n_matrix_mul_iron(...)` path used by `FFN_Linear_1@MPC`

## Working Build Commands

From `he_compiler/EzPC_bolt/EzPC/SCI/build`:

```bash
cmake ..
cmake --build . --target BOLT_GELU_BRIDGE BOLT_SOFTMAX_BRIDGE BOLT_LAYERNORM_BRIDGE BOLT_FFN_LINEAR1_BRIDGE BOLT_QK_MATMUL_MPC_BRIDGE BOLT_ATTN_V_MATMUL_MPC_BRIDGE -j4
```

## Working Validation Commands

From `he_compiler/operator_execution_framework`:

```bash
# standalone bridge checks
python experiments/validate_gelu_bridge.py
python experiments/validate_softmax_bridge.py
python experiments/validate_layernorm_bridge.py
python experiments/validate_ffn_linear1_bridge.py
python experiments/validate_qk_matmul_bridge.py
python experiments/validate_attn_v_matmul_bridge.py
python experiments/validate_gelu_he_nexus_bridge.py
python experiments/validate_softmax_he_nexus_bridge.py
python experiments/validate_ffn_linear1_he_nexus_bridge.py
python experiments/validate_qk_matmul_he_nexus_bridge.py
python experiments/validate_attn_v_matmul_he_nexus_bridge.py

# python wrapper checks
python experiments/validate_gelu_wrapper.py
python experiments/validate_softmax_wrapper.py
python experiments/validate_layernorm_wrapper.py
python experiments/validate_ffn_linear1_wrapper.py
python experiments/validate_qk_matmul_wrapper.py
python experiments/validate_attn_v_matmul_wrapper.py
python experiments/validate_gelu_he_nexus_wrapper.py
python experiments/validate_softmax_he_nexus_wrapper.py
python experiments/validate_layernorm_he_nexus_wrapper.py
python experiments/validate_ffn_linear1_he_nexus_wrapper.py
python experiments/validate_qk_matmul_he_nexus_wrapper.py
python experiments/validate_attn_v_matmul_he_nexus_wrapper.py
python experiments/validate_residual_add_semantic.py
python experiments/validate_ffn_linear2_he_nexus_wrapper.py
python experiments/validate_ffn_linear2_mpc_wrapper.py

# routed checks
python experiments/validate_ffn_linear1_routed.py
python experiments/validate_attention_matmul_routed.py
python experiments/validate_gelu_he_nexus_routed.py
python experiments/validate_softmax_he_nexus_routed.py
python experiments/validate_layernorm_he_nexus_routed.py
python experiments/validate_ffn_linear1_he_nexus_routed.py
python experiments/validate_attention_matmul_he_nexus_routed.py
python experiments/validate_semantic_ops_routed.py
python experiments/validate_he_nexus_capabilities.py
python experiments/run_experiment.py --config experiments/configs/ffn_linear1_mpc_only.json --batch 1 --seq 2 --hidden 4
```

## Known Limitations / Unresolved Issues

- SCI startup is sometimes flaky due to port-block binding races (`Address already in use`) and occasional process timeouts.
  - `Softmax`, `LayerNorm`, `FFN_Linear_1`, `Attention_QK_MatMul`, and `Attention_V_MatMul` wrappers include retry logic with new port blocks.
  - `GeLU` currently has limited retry behavior and may require rerun on transient bind conflicts.
- Real MPC validations that reuse SCI bridges can still hang or take a long time in some environments.
- `FFN_Linear_1` wrapper currently enforces `n = B*S <= 64` because bert_bolt `NonLinear` uses fixed internal thread/pack arrays.
- `Attention_V_MatMul` is compatibility-first:
  - internal canonical compute shape is `[B,H,S,D]`
  - default output remains router-compatible `[B,S,H*D]`
  - optional canonical output can be enabled via context flag `attention_return_canonical`
- Compiler prototype is still standalone and not yet wired into runtime router dispatch.

## Current Routing Model

- Operator-first, method-aware model:
  - `Operator` directories contain method implementations.
  - Runtime currently routes by backend.
  - Method-level dispatch is the intended planning/compilation contract.

## Compiler Prototype Status: `compiler/min_cut`

- Current state:
  - runnable binary-domain (`HE`/`MPC`) min-cut placement prototype
  - includes plan builder, baseline comparison, and figure generation
- Primary entry commands:
  - `python -m compiler.min_cut.demo`
  - `python -m compiler.min_cut.compiler_figure`
- Detailed function-level reference is preserved in `docs/architecture.md`

## Recommended Next Milestone

Implement explicit method-level dispatch in runtime (for example `op -> method -> backend`) and then integrate **`Out_Projection.method_mpc_bolt @ MPC`** as the next real linear method using the existing staged workflow:

- bridge validation
- wrapper validation
- routed validation
- capability matrix + registry update

After that, the next honest high-value gaps remain:

- `Linear_QKV`
- `Out_Projection`
- possibly `Embedding`, but only if a real backend-specific primitive path can be added honestly

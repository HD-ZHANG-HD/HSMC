# Step 2/3 - Refactoring Architecture and Extraction Plan

## Target Structure

- `framework/types.py`
  - `BackendType`, `TensorValue`, `ExecutionContext`.
- `framework/operators.py`
  - operator specs + `OperatorRegistry`.
- `framework/router.py`
  - runtime backend selection and automatic conversion insertion.
- `framework/backends.py`
  - backend-specific operator implementations.
- `framework/adapters.py`
  - explicit path mapping to existing MPC/HE-MPC sources.

## Operator Interface

The runtime executes operators through the unified abstraction:

- logical interface:
  - `execute(op_name, input_tensors, backend_type)`
- implemented by:
  - registry lookup + backend function call
  - router decides `backend_type` from JSON config at runtime

## Backend Routing and Conversion

For each operator in the BERT sequence:

1. Load `backend_type` from config (`MPC`, `HE`, `HYBRID`).
2. Convert each input tensor domain to `backend_type` if needed:
   - `HE -> MPC`: `HE_to_MPC`
   - `MPC -> HE`: `MPC_to_HE`
   - transitions to/from `HYBRID` also tracked explicitly.
3. Dispatch implementation from `OperatorRegistry`.
4. Save output as typed tensor with resulting domain.

## Extraction Guidance from Existing Code

- Extract HE linear kernels from `linear.cpp` into reusable wrappers:
  - `linear_1` -> `Linear_QKV_HE`
  - `linear_2` variants -> `Out_Projection_HE`, `FFN_Linear_1_HE`, `FFN_Linear_2_HE`
- Extract MPC nonlinear kernels from `nonlinear.cpp`:
  - `softmax`, `gelu`, `layer_norm`
- Extract conversion helpers from `bert.cpp`:
  - `he_to_ss_*` and `ss_to_he_*` into dedicated conversion operator modules.

## Minimal-Intrusion Principle

- Keep existing codebases unchanged where possible.
- Add thin wrappers/adapters that call current kernels.
- Place all orchestration and operator routing under `he_compiler/operator_execution_framework`.

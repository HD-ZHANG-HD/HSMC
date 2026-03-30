# Operator Execution Framework Architecture

## Goal

Build an operator-first execution framework for transformer/BERT inference where:

- operators are modular units,
- backend/method placement is determined by a compiler-driven hybrid planning pipeline,
- backend transitions are inserted explicitly in a compiler-generated execution plan,
- and method-level operator dispatch is supported by design.

## Project Root

- Framework root: `he_compiler/operator_execution_framework`
- Legacy implementations reused:
  - `he_compiler/MPCFormer/src/benchmark/models.py`
  - `he_compiler/EzPC_bolt/EzPC/SCI/tests/bert_bolt/{bert.cpp,linear.cpp,nonlinear.cpp}`

## Documentation Consolidation Note

This file consolidates architecture, operator design, restricted HE/NEXUS contract, and compiler prototype reference material that was previously spread across multiple markdown files.

The goal is to keep persistent design information in a smaller number of source-of-truth files without dropping prior content.

## Modified: Target Pipeline

Target data flow:

`IR -> Cost Model -> Planner -> Execution Plan -> Runtime`

Target layered architecture:

1. IR Layer
2. Cost Model Layer
3. Planner Layer
4. Execution Plan Layer
5. Runtime Layer
6. Operator Layer
7. Backend / Bridge Layer

Important architecture rule:

- compiler determines where backend/domain transitions occur
- runtime executes a compiler-generated plan
- operators receive tensors already in the expected execution domain
- backend/bridge layers implement concrete execution methods

Current mismatch:

- today, the compiler-side path exists only as a standalone prototype inside `compiler/min_cut`
- runtime still performs backend-routed execution directly in `runtime/router.py`
- the target compiler-driven hybrid PPML pipeline is therefore only partially reflected in the current implementation

## Modified: Layered Architecture

### 1) IR Layer (`ir/`) [Status: planned]

Defines backend-independent transformer computation.

- `operator_schema.py` (planned): canonical operator schema definitions
- `bert_graph.py` (planned): transformer graph representation

IR must not depend on backend-specific integrations.

Current implementation note:

- IR is defined conceptually, but the main framework does not yet have an implemented compiler-facing IR as the source of truth for planning and execution.

### 2) Cost Model Layer (`compiler/min_cut/profiler_db.py`, `compiler/min_cut/cost_model.py`) [Status: partially implemented]

Estimates operator and conversion latency for backend/domain choices.

Target responsibilities:

- load profiler data
- estimate operator cost under `HE` / `MPC`
- estimate conversion cost across domains
- provide cost signals to the planner

Current implementation:

- `compiler/min_cut/profiler_db.py`
  - implemented for synthetic/profile JSON ingestion and lookup
- `compiler/min_cut/cost_model.py`
  - implemented for exact/nearest/linear/size-scaling estimation

Current mismatch:

- cost modeling exists inside `compiler/min_cut`, but it is not integrated into the main framework pipeline
- runtime does not currently query this layer before deciding execution behavior

### 3) Planner Layer (`compiler/min_cut/domain_assignment.py`) [Status: prototype (standalone, not integrated)]

Chooses backend/domain placement for operators based on cost-model outputs.

Target responsibilities:

- choose where `HE` vs `MPC` execution should occur
- minimize total latency including conversion costs
- produce domain assignments suitable for executable planning

Current implementation:

- `compiler/min_cut/domain_assignment.py`
  - contains graph representation, min-cut construction, and assignment logic

Current mismatch:

- the planner exists only as a standalone prototype
- it is not wired into runtime execution
- runtime still makes backend decisions directly from backend config/routing instead of from planner output

### 4) Execution Plan Layer (`compiler/min_cut/plan_builder.py`) [Status: partially implemented]

Converts planner output into an explicit execution plan.

Target responsibilities:

- linearize execution order
- insert explicit conversion steps on cross-domain edges
- produce a plan that runtime can execute directly

Current implementation:

- `compiler/min_cut/plan_builder.py`
  - builds ordered execution steps and explicit conversion steps
- `compiler/min_cut/demo.py`
  - prints plans for standalone prototype cases

Current mismatch:

- plan-building logic exists, but runtime does not consume a compiler-generated plan object
- current runtime executes operator sequence directly instead of executing a precomputed plan

### 5) Runtime Layer (`runtime/`) [Status: implemented]

Owns execution orchestration and plan execution.

Target responsibilities:

- execute a compiler-generated execution plan
- invoke operators and conversions exactly as specified by that plan
- remain agnostic to backend-placement decision logic

Current implementation:

- `runtime/types.py`
  - `BackendType`, `TensorValue`, `ExecutionContext`
- `runtime/operator_specs.py`
  - canonical BERT operator sequence
- `runtime/operator_registry.py`
  - registry of executable operator implementations
- `runtime/router.py`
  - pipeline execution and domain conversion insertion
- `runtime/capabilities.py`
  - queryable capability status registry (`real-integrated` / `restricted-integrated` / `mock` / `unsupported`)
- `runtime/conversion/`
  - runtime-level HE<->MPC conversion registry, manager, capabilities, and directional conversion methods

Current mismatch:

- runtime is implemented, but today it still performs backend-routed execution through `runtime/router.py`
- runtime therefore still participates in routing/domain decision behavior that should belong to compiler/planner layers in the target architecture
- conversion execution is implemented in runtime, but conversion placement is still determined by runtime routing rather than by a precomputed compiler plan

### 6) Operator Layer (`operators/`) [Status: partially implemented]

Primary abstraction for execution units. Each operator has its own directory.

Current extracted operators:

- `operators/gelu/`
- `operators/softmax/`
- `operators/layernorm/`
- `operators/linear_ffn1/`
- `operators/ffn_linear_2/`
- `operators/residual_add/`
- `operators/attention_qk_matmul/`
- `operators/attention_v_matmul/`

Each operator directory contains:

- `spec.py`: operator metadata (inputs, outputs, attributes)
- method modules (for example `method_mpc_bolt.py`)

This structure is method-dispatch ready and supports multiple implementations per operator.

Current mismatch:

- operator structure is in place, but not all semantic BERT operators have real or restricted-integrated backend methods yet
- method-level dispatch exists structurally, but compiler/runtime do not yet select methods via an integrated planning flow

### 7) Backend / Bridge Layer (`backends/`, `bridge/`) [Status: partially implemented]

Houses backend-specific integration code and external dependency bindings (for example MPC/HE stacks). This layer may depend on external repositories.

Important backend adapter modules currently used:

- `backends/he_nexus_attention_adapter.py`
- `backends/he_nexus_linear_ffn1_adapter.py`
- `backends/he_nexus_layernorm_adapter.py`

Bridge sublayer:

Contains bridge code that interfaces the framework with external runtimes.

Existing C++ bridge sources currently live in legacy path `bridge/` and map to:

- `gelu_bridge.cpp` -> `NonLinear::gelu`
- `softmax_bridge.cpp` -> `NonLinear::softmax`
- `layernorm_bridge.cpp` -> `NonLinear::layer_norm`
- `ffn_linear1_bridge.cpp` -> `NonLinear::n_matrix_mul_iron` (+ bias add)
- `FFN_Linear_2@MPC` currently reuses the same `ffn_linear1_bridge.cpp` execution primitive semantically
- `qk_matmul_mpc_bridge.cpp` -> `NonLinear::n_matrix_mul_iron`
- `attn_v_matmul_mpc_bridge.cpp` -> `NonLinear::n_matrix_mul_iron`
- `Residual_Add` currently lowers to backend-native tensor add and does not require a separate external bridge

Built under SCI as:

- `BOLT_GELU_BRIDGE`
- `BOLT_SOFTMAX_BRIDGE`
- `BOLT_LAYERNORM_BRIDGE`
- `BOLT_FFN_LINEAR1_BRIDGE`
- `BOLT_QK_MATMUL_MPC_BRIDGE`
- `BOLT_ATTN_V_MATMUL_MPC_BRIDGE`

Current mismatch:

- several backend/bridge paths are real or restricted-integrated, but the full BERT operator set is not yet covered by honest executable methods
- bridge execution exists for selected MPC operators, while many semantic operators still rely on mock or lowered paths

## Current Compatibility Strategy

To preserve behavior during migration, `framework/` modules currently act as compatibility facades that re-export runtime/operator implementations. Existing scripts can still import `framework.*` while new code should prefer `runtime.*` and `operators.*`.

## Operator-Method-Backend-Bridge Design

## Purpose

This section defines how operator execution is organized in the framework and how to reason about implementation variants.

## Core Entities

- **Operator**
  - Logical transformer computation unit (for example `Softmax`, `GeLU`, `LayerNorm`).
  - Represented in runtime sequence by `runtime/operator_specs.py`.
  - Operator metadata is stored in `operators/<op>/spec.py`.

- **Method**
  - Concrete implementation variant of an operator.
  - Lives in `operators/<op>/method_*.py`.
  - Example method names: `method_mpc_bolt`, `method_runtime_default`.

- **Backend**
  - Execution domain (`MPC`, `HE`, `HYBRID`).
  - In the target architecture, planner/compiler chooses backend placement and runtime executes that decision.
  - In current code, runtime still routes by backend in `runtime/router.py`.

- **Bridge**
  - External runtime interface used by a method when calling non-Python kernels.
  - Current real MPC methods call SCI bridge binaries built from C++ bridge sources.

## Relationship Model

At design level:

1. Compiler/planner chooses an `Operator`.
2. Compiler/planner chooses a `Method` for that operator.
3. Compiler/planner emits an execution plan with explicit backend placement and conversions.
4. Runtime executes that plan.
5. Method executes on a `Backend`.
6. If external kernels are required, method uses a `Bridge`.

Current code state:

- Operator directories and methods are implemented.
- Runtime dispatch is backend-keyed today.
- Method-level dispatch is the next runtime/compiler milestone.

## Current Examples

### GeLU

- Operator: `GeLU`
- Method: `operators/gelu/method_mpc_bolt.py`
- Backend: `MPC`
- Bridge binary: `BOLT_GELU_BRIDGE`
- External primitive: `NonLinear::gelu(...)`

### Softmax

- Operator: `Softmax`
- Method: `operators/softmax/method_mpc_bolt.py`
- Backend: `MPC`
- Bridge binary: `BOLT_SOFTMAX_BRIDGE`
- External primitive: `NonLinear::softmax(...)`

### LayerNorm

- Operator: `LayerNorm`
- Method: `operators/layernorm/method_mpc_bolt.py`
- Backend: `MPC`
- Bridge binary: `BOLT_LAYERNORM_BRIDGE`
- External primitive: `NonLinear::layer_norm(...)`

### LayerNorm (Restricted HE / NEXUS)

- Operator: `LayerNorm`
- Method: `operators/layernorm/method_he_nexus.py`
- Backend: `HE`
- Adapter: `backends/he_nexus_layernorm_adapter.py`
- External primitive mapping: `NEXUS/src/layer_norm.cpp -> LNEvaluator::layer_norm(...)`
- Status: `restricted-integrated`
- Contract summary:
  - input `[B,S,768]`
  - `1 <= B*S <= 16`
  - affine `weight` / `bias` unsupported
  - output `[B,S,768]`
  - semantics follow the current NEXUS execution style rather than a fully general framework LayerNorm

### Residual_Add

- Operator: `Residual_Add`
- Method: `operators/residual_add/method_runtime_default.py`
- Backend: `MPC` / `HE`
- Primitive mapping: backend-native tensor add
- Status:
  - `MPC`: `real-integrated`
  - `HE`: `real-integrated`
- Design note:
  - this is a semantic operator-level implementation
  - trace keeps the semantic operator name and also records the add lowering explicitly
  - no routing or conversion logic is hidden inside the operator

### FFN_Linear_2

- Operator: `FFN_Linear_2`
- Method: `operators/ffn_linear_2/method_mpc_bolt.py`
- Backend: `MPC`
- Lowered primitive: `NonLinear::n_matrix_mul_iron(...)`
- Lowering rule:
  - reuses the existing `FFN_Linear_1.method_mpc_bolt` execution path
  - keeps semantic `FFN_Linear_2` trace visibility

### FFN_Linear_2 (Restricted HE / NEXUS)

- Operator: `FFN_Linear_2`
- Method: `operators/ffn_linear_2/method_he_nexus.py`
- Backend: `HE`
- Lowered primitive: `NEXUS/src/matrix_mul.cpp -> MMEvaluator::matrix_mul(...)`
- Lowering rule:
  - reuses the existing NEXUS linear adapter path behind `FFN_Linear_1.method_he_nexus`
  - keeps semantic `FFN_Linear_2` trace visibility
- Status: `restricted-integrated`
- Contract summary:
  - configurable linear map `[B,S,H] -> [B,S,O]`
  - bounded by the existing restricted NEXUS linear adapter token limits
  - should be treated as restricted HE support, not general unrestricted HE linear support

## Where to Update When Adding a New Method

For a new method integration:

1. Add/update operator metadata in `operators/<op>/spec.py`.
2. Add method module `operators/<op>/method_<name>.py`.
3. Register method/backend execution path in runtime registration flow (`framework/backends.py` currently).
4. Update capability status (`runtime/capabilities.py` and `docs/backend_capability_matrix.md`).
5. Add staged validation scripts:
   - bridge standalone
   - wrapper
   - routed execution

## Method-Level Execution Model

Conceptual model:

- `Operator`: logical transformer operation (for example `Softmax`)
- `Method`: concrete implementation variant (for example `method_mpc_bolt`)
- `Backend`: execution domain used by that method (`MPC`, `HE`, `HYBRID`)
- `Bridge`: external integration boundary used by a method when needed

Current state in code:

- runtime routing key is currently backend-centric (`op -> backend`) in `runtime/router.py`,
- operator directories already define per-operator methods and `spec.py`,
- real MPC methods are implemented as `method_mpc_bolt.py` for:
  - `GeLU`
  - `Softmax`
  - `LayerNorm`
  - `FFN_Linear_1`
  - `FFN_Linear_2` (by semantic lowering to the existing FFN linear primitive)
- real MPC methods are implemented as `method_mpc.py` for:
  - `Attention_QK_MatMul`
  - `Attention_V_MatMul`
- real semantic add methods are implemented as `method_runtime_default.py` for:
  - `Residual_Add`
- restricted NEXUS HE methods are implemented as `method_he_nexus.py` for:
  - `LayerNorm`
  - `FFN_Linear_1`
  - `FFN_Linear_2`
  - `Attention_QK_MatMul`
  - `Attention_V_MatMul`

Near-term direction:

- keep runtime execution contract stable while shifting placement decisions upward into compiler/planner layers,
- add explicit method selector (`op -> method`) during planning/compilation,
- support multiple methods per operator (for example MPC bolt, MPC alt, HE polynomial).

## Restricted HE / NEXUS Contracts

This section preserves the detailed contract information formerly kept in separate contract documents.

## Attention HE NEXUS Contract

### Scope

This section unifies the restricted-integration contracts for:

- `Attention_QK_MatMul.method_he_nexus @ HE`
- `Attention_V_MatMul.method_he_nexus @ HE`

Both methods are **restricted-integrated**. They are usable only within the explicit input/layout regime listed here.

### Design Intent

These adapters reuse NEXUS matrix-multiplication style internals via framework-side adapter code, but they do not expose a fully general-purpose HE attention API yet.

NEXUS source mapping:

- `he_compiler/NEXUS/src/matrix_mul.cpp`
  - `MMEvaluator::matrix_mul`
  - `MMEvaluator::enc_compress_ciphertext`
  - `MMEvaluator::expand_ciphertext`
- `he_compiler/NEXUS/src/main.cpp` MM packing workflow

Framework adapter bridge:

- `backends/he_nexus_attention_adapter.py`

### Unified Restricted Contract

- Hidden size fixed: `H = 768`
- Heads fixed: `num_heads = 12`
- Head dim fixed: `D = 64`
- Batch bound: `1 <= B <= 8`
- Sequence bound: `1 <= S <= 128`
- QKV layout for HE attention adapters: packed tensor `[3, B, S, 768]`

If input does not satisfy this contract, methods must fail explicitly (no silent fallback, no over-claim).

### Operator Contracts

#### Attention_QK_MatMul

- Method: `operators/attention_qk_matmul/method_he_nexus.py`
- Input: one tensor `qkv_out` in packed layout `[3, B, S, 768]`
- Output: `qk_scores` with shape `[B, 12, S, S]`
- Status: `restricted-integrated`

#### Attention_V_MatMul

- Method: `operators/attention_v_matmul/method_he_nexus.py`
- Inputs:
  - `attn_probs`: `[B, 12, S, S]`
  - packed `qkv_out`: `[3, B, S, 768]`
- Output:
  - default: `[B, S, 768]`
  - optional canonical (`attention_return_canonical=True`): `[B, 12, S, 64]`
- Status: `restricted-integrated`

### Supported / Unsupported Shape Table

| Operator | Input Shape(s) | Output Shape | Status |
|---|---|---|---|
| Attention_QK_MatMul | `[3,B,S,768]`, with `1<=B<=8`, `1<=S<=128` | `[B,12,S,S]` | supported |
| Attention_QK_MatMul | any non-packed input (for example `[B,S,768]`, `[B,12,S,64]` pair) | N/A | unsupported |
| Attention_QK_MatMul | packed but `H!=768` (for example `[3,B,S,512]`) | N/A | unsupported |
| Attention_QK_MatMul | packed but heads not 12 / out-of-bound `B` or `S` | N/A | unsupported |
| Attention_V_MatMul | `[B,12,S,S]` + `[3,B,S,768]`, with `1<=B<=8`, `1<=S<=128` | `[B,S,768]` (or canonical `[B,12,S,64]`) | supported |
| Attention_V_MatMul | `attn_probs` not `[B,12,S,S]` | N/A | unsupported |
| Attention_V_MatMul | qkv not packed `[3,B,S,768]` | N/A | unsupported |
| Attention_V_MatMul | any case violating bounds or fixed hidden/head contract | N/A | unsupported |

### Notes

- This contract is intentionally strict and honest. It prevents accidental claims of general HE attention support.
- Future work can relax constraints by adding a generalized NEXUS bridge for batched/head-aware matmul and broader packing support.

## LayerNorm HE NEXUS Contract

### Scope

This section defines the restricted-integration contract for:

- `LayerNorm.method_he_nexus @ HE`

This method is **restricted-integrated**. It is usable only within the explicit shape and parameter regime listed here.

### Design Intent

This adapter follows the current NEXUS layer-norm execution style instead of pretending to provide a fully general HE LayerNorm API.

NEXUS source mapping:

- `he_compiler/NEXUS/src/layer_norm.cpp`
  - `LNEvaluator::layer_norm`
- `he_compiler/NEXUS/src/main.cpp`
  - current demo path packs a 768-feature vector with `len=1024`

Framework adapter bridge:

- `backends/he_nexus_layernorm_adapter.py`

### Restricted Contract

- Input tensor shape fixed to `[B, S, 768]`
- Hidden size fixed: `H = 768`
- Flattened token bound: `1 <= B*S <= 16`
- Packed execution length fixed to the current NEXUS-style assumption: `packed_len = 1024`
- Affine `weight` / `bias` are unsupported
- Output shape preserved: `[B, S, 768]`

If input or runtime assumptions do not satisfy this contract, the method must fail explicitly.

### Semantic Note

This path follows the current NEXUS layer-norm computation style from `LNEvaluator::layer_norm`, which normalizes by the inverse root-mean-square over the last dimension.

That means this method should be treated as a NEXUS-restricted execution path, not as a fully general framework LayerNorm implementation.

### Operator Contract

- Method: `operators/layernorm/method_he_nexus.py`
- Input: one tensor `attn_residual` with shape `[B, S, 768]`
- Output: `attn_norm` with shape `[B, S, 768]`
- Status: `restricted-integrated`

### Supported / Unsupported Shape Table

| Operator | Input Shape / Params | Output Shape | Status |
|---|---|---|---|
| LayerNorm | `[B,S,768]`, with `1 <= B*S <= 16`, no affine params | `[B,S,768]` | supported |
| LayerNorm | any input with `H != 768` | N/A | unsupported |
| LayerNorm | any input with `B*S > 16` | N/A | unsupported |
| LayerNorm | affine `weight` provided | N/A | unsupported |
| LayerNorm | affine `bias` provided | N/A | unsupported |

### Validation Anchors

- `python experiments/validate_layernorm_he_nexus_wrapper.py`
- `python experiments/validate_layernorm_he_nexus_routed.py`
- `python experiments/validate_he_nexus_capabilities.py`

### Notes

- This contract is intentionally strict and honest.
- Future work can relax constraints only if the NEXUS path is generalized and the framework-side method is updated to match that broader contract.

## Modified: Planner Layer Integration Of The Existing Compiler Prototype

The existing `compiler/min_cut` module should be understood conceptually as part of the Planner stack in the target architecture, not as a separate side project.

Current status:

- prototype (standalone, not integrated)

Current scope:

- profiler-driven latency estimation
- graph domain assignment
- execution plan expansion with conversion steps
- baseline and visualization tooling

Current mismatch:

- this prototype is currently standalone and not yet wired into runtime execution

## Min-Cut Compiler Prototype Reference

## Purpose

This section is the operational reference for the current standalone planner prototype in `compiler/min_cut`.
It is written so a new agent can continue development without reading source files first.

The module provides a binary domain placement compiler prototype for operator graphs:

- Domain choices: `HE` or `MPC`
- Objective: minimize total latency
  - operator execution cost
  - plus cross-domain conversion cost
- Solver: s-t min-cut

## Scope and File Responsibilities

- `compiler/min_cut/profiler_db.py`
  - In-memory microbenchmark database and shape-indexed lookups.
- `compiler/min_cut/cost_model.py`
  - Latency estimation logic with multiple interpolation/fallback strategies.
- `compiler/min_cut/domain_assignment.py`
  - Graph data model, min-cut construction/solve, and assignment cost evaluation.
- `compiler/min_cut/plan_builder.py`
  - Converts assignment into executable steps and computes baseline totals.
- `compiler/min_cut/demo.py`
  - End-to-end case runner and textual result printer.
- `compiler/min_cut/compiler_figure.py`
  - Batch visualization of baseline vs optimized latency per case.
- `compiler/min_cut/test/*.json`
  - Synthetic profiler and graph inputs, plus case list.

## Data Contracts

### 1) Profiler JSON

Expected top-level keys:

- `records`: operator benchmark records
- `conversion_records`: domain conversion benchmark records

Operator record fields:

- `op_type` (string)
- `domain` (`HE` or `MPC`)
- `input_shape` (list[int])
- `output_shape` (list[int])
- `latency_ms` (float)
- `metadata` (optional object)

Conversion record fields:

- `from_domain` (`HE` or `MPC`)
- `to_domain` (`HE` or `MPC`)
- `tensor_shape` (list[int])
- `latency_ms` (float)

### 2) Graph JSON

Expected top-level keys:

- `graph_id` (optional string)
- `nodes` (list)
- `edges` (list)

Node fields:

- `node_id` (string)
- `op_type` (string)
- `input_shape` (list[int])
- `output_shape` (list[int])

Edge fields:

- `src` (string node id)
- `dst` (string node id)
- `tensor_shape` (list[int])

### 3) Case JSON (`test/cases.json`)

Each case item includes:

- `name`
- `description`
- `profiler_json`
- `graph_json`

## Function-Level Reference

### `profiler_db.py`

#### Core role

Loads profiler payloads, normalizes shape values, and provides exact/candidate lookup APIs.

#### Functions and classes

- `Domain`
  - Type alias restricted to `HE` / `MPC`.
- `_as_shape(value)`
  - Converts iterable shape to canonical tuple of ints.
- `BenchmarkRecord`
  - Immutable operator benchmark row.
- `ConversionRecord`
  - Immutable conversion benchmark row.
- `ProfilerDB(records, conversion_records)`
  - Builds internal indexes:
    - operator index by `(op_type, domain)`
    - conversion index by `(from_domain, to_domain)`
- `ProfilerDB.from_json(json_path)`
  - Reads JSON and delegates to `from_dict`.
- `ProfilerDB.from_dict(payload)`
  - Validates domain values and builds typed records.
- `ProfilerDB.get_operator_records(op_type, domain)`
  - Returns candidate list for estimation.
- `ProfilerDB.find_exact_operator_record(op_type, domain, input_shape, output_shape)`
  - Exact tuple-shape match or `None`.
- `ProfilerDB.get_conversion_records(from_domain, to_domain)`
  - Candidate conversion records.
- `ProfilerDB.find_exact_conversion_record(from_domain, to_domain, tensor_shape)`
  - Exact conversion match or `None`.

#### Extension guidance

- Add new metadata fields only if they are backward-compatible.
- Keep domain validation strict; this prevents silent bad records.

### `cost_model.py`

#### Core role

Estimates operator/conversion latency from `ProfilerDB`.
Supports strict and interpolated modes for missing shapes.

#### Functions and classes

- `EstimationStrategy`
  - One of: `exact`, `nearest`, `linear`, `size_scaling`, `auto`.
- `_numel(shape)`
  - Product of dimensions.
- `_shape_distance(a_in, a_out, b_in, b_out)`
  - Distance metric used by nearest-neighbor fallback.
- `CostEstimate`
  - Result object with:
    - `latency_ms`
    - `strategy_used`
- `CostModel(db, default_strategy="auto")`
  - Stateful estimator using profiler data.
- `CostModel.estimate_node_cost(op_type, domain, input_shape, output_shape, strategy=None)`
  - Behavior:
    1. Try exact match first.
    2. If not found, apply requested/default strategy.
    3. Raise if no candidates exist.
- `CostModel.estimate_conversion_cost(tensor_shape, from_domain, to_domain, strategy=None)`
  - Special handling:
    - same domain => zero latency
    - otherwise exact or strategy fallback
- `CostModel._nearest(...)`
  - Chooses closest benchmark by shape-distance.
- `CostModel._size_scaling(...)`
  - Scales nearest latency by size ratio.
- `CostModel._linear_fit(...)`
  - Fits linear model from candidate points.
- `CostModel._fit_predict(xs, ys, xq)`
  - Shared least-squares predictor.

#### Extension guidance

- If adding a new strategy, wire both node and conversion paths.
- Preserve exact-first policy for reproducibility.
- Keep non-negative latency clamp in extrapolation paths.

### `domain_assignment.py`

#### Core role

Defines graph structures, computes min-cut assignment, and reports total cost.

#### Data structures

- `OperatorNode`
  - `node_id`, `op_type`, `input_shape`, `output_shape`.
- `DataEdge`
  - `src`, `dst`, `tensor_shape`.
- `OperatorGraph`
  - `graph_id`, `nodes`, `edges`.
- `AssignmentResult`
  - `assignment`
  - `node_cost_ms`
  - `conversion_cost_ms`
  - `total_cost_ms`
  - `per_node_costs`

#### Functions

- `_as_shape(values)`
  - Shape normalization helper.
- `load_graph_json(path)`
  - Parses graph JSON into typed objects.
- `_add_capacity(capacity, u, v, cap)`
  - Adds edge capacity and residual reverse entry.
- `_edmonds_karp_min_cut(capacity, source, sink)`
  - Computes max-flow/min-cut and source-reachable set.
- `evaluate_assignment_cost(graph, assignment, cost_model)`
  - Computes:
    - total operator cost under assignment
    - total cross-domain conversion cost
    - total latency
- `assign_domains_min_cut(graph, cost_model)`
  - Main compiler step.
  - Construction:
    - source side interpreted as `HE`
    - sink side interpreted as `MPC`
    - `source -> node` capacity uses node `MPC` cost
    - `node -> sink` capacity uses node `HE` cost
    - between connected nodes, add symmetric disagreement penalty
      from mean conversion cost:
      `0.5 * (HE->MPC + MPC->HE)`
  - Output: domain map and cost summary.
- `make_uniform_assignment(graph, domain)`
  - Utility baseline assignment (`all HE` or `all MPC`).

#### Extension guidance

- Keep graph acyclic expectation in downstream plan builder.
- If introducing directional conversion penalties in cut graph,
  document the math because current implementation is symmetric.

### `plan_builder.py`

#### Core role

Transforms graph + assignment into a linear execution plan with explicit conversion steps.

#### Symbols

- `NONLINEAR_OPS`
  - Nonlinear operator set used by hybrid baseline.
- `PlanStep`
  - Lightweight typed structure; output plan currently uses dict steps.

#### Functions

- `_topological_order(graph)`
  - Kahn-style topological sort.
  - Raises if cycle is detected.
- `build_execution_plan(graph, assignment, cost_model, include_baselines=True)`
  - For each node in topological order:
    1. inserts conversion steps for incoming edges crossing domains
    2. inserts operator execution step
  - Computes `cost_breakdown` via `evaluate_assignment_cost`.
  - Optional baseline totals:
    - `all_he_total_ms`
    - `all_mpc_total_ms`
    - `hybrid_linear_he_nonlinear_mpc_total_ms`

#### Extension guidance

- Keep output plan format stable if used by downstream tooling.
- If adding new baseline types, add them as additive fields, not replacements.

### `demo.py`

#### Core role

Case-driven CLI runner for validation and reporting.

#### Functions

- `_print_case_header(case)`
  - Prints case metadata.
- `_print_node_costs(result)`
  - Prints per-node HE/MPC estimated costs.
- `_print_assignment(result)`
  - Prints chosen domain per node.
- `_print_plan(plan)`
  - Prints step list, cost breakdown, and baselines.
- `run_demo()`
  - Loads all cases from `test/cases.json`.
  - For each case:
    1. load profiler + graph
    2. run min-cut assignment
    3. build execution plan with baselines
    4. check `optimized <= all_HE` and `optimized <= all_MPC`
  - Fails process if any case violates baseline check.

#### Operational use

- Quick smoke check after algorithm or cost-model changes.
- First command to run before changing benchmark/test assets.

### `compiler_figure.py`

#### Core role

Produces grouped bar chart comparing baseline totals and min-cut optimized total across all cases.

#### Functions

- `_collect_case_times(case, base_dir)`
  - Runs one case and returns:
    - `optimized`
    - `all_he`
    - `all_mpc`
    - `hybrid`
- `generate_figure(output_file="compiler_case_times.png")`
  - Loads all cases and renders grouped bars.
  - Saves PNG into `compiler/min_cut`.
- `main()`
  - CLI entrypoint that writes image and prints output path.

#### Operational use

- Regression visualization after changing:
  - cost strategy behavior
  - conversion penalties
  - test case set

## Development Workflow for New Agents

1. Run baseline sanity:
   - `python -m compiler.min_cut.demo`
2. Generate figure:
   - `python -m compiler.min_cut.compiler_figure`
3. If changing estimation behavior:
   - update `cost_model.py`
   - validate all cases still pass
   - compare old/new chart deltas
4. If changing assignment logic:
   - update `domain_assignment.py`
   - verify cut-side interpretation remains consistent (`HE` source side, `MPC` sink side)
5. If changing plan format:
   - update `plan_builder.py`
   - confirm `demo.py` and plotting script still parse outputs

## Known Constraints and Risks

- Current solver is Edmonds-Karp; complexity is fine for small synthetic graphs but may not scale for large production graphs.
- Conversion penalty in cut construction is symmetric average; it does not preserve direction-specific asymmetry.
- Baseline comparison in demo currently enforces better-than-all-HE and better-than-all-MPC; it does not enforce better-than-hybrid.

## New: Current Implementation Status Summary

| Layer | Target Responsibility | Status |
|---|---|---|
| IR Layer | backend-independent transformer computation | planned |
| Cost Model Layer | estimate operator/conversion cost for planning | partially implemented |
| Planner Layer | choose backend/domain placement | prototype (standalone, not integrated) |
| Execution Plan Layer | build explicit executable plan with conversions | partially implemented |
| Runtime Layer | execute compiler-generated execution plan | implemented |
| Operator Layer | semantic operator methods and metadata | partially implemented |
| Backend / Bridge Layer | backend-specific kernels, adapters, and bridges | partially implemented |

## What Is Still Missing To Reach The Target Architecture

- IR must be implemented as the main frontend representation consumed by the compiler path.
- Cost model, planner, and execution-plan builder must be integrated into the main architecture instead of remaining inside standalone `compiler/min_cut`.
- Runtime must stop making backend-placement decisions itself and instead execute a compiler-produced execution plan.
- Method selection must move from backend-keyed routing toward compiler-selected `operator -> method -> backend`.
- The full semantic BERT operator set still needs honest executable backend coverage, especially for operators that remain mock.

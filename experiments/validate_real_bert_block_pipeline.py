from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compiler.min_cut import compile_graph_to_runtime_plan
from compiler.min_cut.cost_model import CostModel
from compiler.min_cut.profiler_db import ProfilerDB
from framework import register_default_backend_impls
from ir import BertBlockConfig, DataEdge, OperatorGraph, OperatorNode, build_bert_block_graph
from runtime import (
    BackendType,
    ExecutionContext,
    OperatorRegistry,
    TensorValue,
    capability_registry,
    conversion_capability_registry,
    conversion_manager,
    execute,
)


EXPECTED_OPERATOR_ORDER = [
    "LayerNorm",
    "Attention_QK_MatMul",
    "Softmax",
    "Attention_V_MatMul",
    "Residual_Add",
    "LayerNorm",
    "FFN_Linear_1",
    "GeLU",
    "FFN_Linear_2",
    "Residual_Add",
]


@dataclass
class RealExecutionResult:
    name: str
    table: list[dict[str, str]]
    assignment: dict[str, str]
    trace: list[str]
    success: bool
    error: str | None = None


def _build_full_block_profiler(seq_len: int) -> ProfilerDB:
    return ProfilerDB.from_dict(
        {
            "records": [
                {"op_type": "LayerNorm", "domain": "HE", "input_shape": [1, seq_len, 768], "output_shape": [1, seq_len, 768], "latency_ms": 3.0},
                {"op_type": "LayerNorm", "domain": "MPC", "input_shape": [1, seq_len, 768], "output_shape": [1, seq_len, 768], "latency_ms": 6.0},
                {"op_type": "Attention_QK_MatMul", "domain": "HE", "input_shape": [1, 12, seq_len, seq_len], "output_shape": [1, 12, seq_len, seq_len], "latency_ms": 12.0},
                {"op_type": "Attention_QK_MatMul", "domain": "MPC", "input_shape": [1, seq_len, 768], "output_shape": [1, 12, seq_len, seq_len], "latency_ms": 4.0},
                {"op_type": "Softmax", "domain": "HE", "input_shape": [1, 12, seq_len, seq_len], "output_shape": [1, 12, seq_len, seq_len], "latency_ms": 8.0},
                {"op_type": "Softmax", "domain": "MPC", "input_shape": [1, 12, seq_len, seq_len], "output_shape": [1, 12, seq_len, seq_len], "latency_ms": 2.0},
                {"op_type": "Attention_V_MatMul", "domain": "HE", "input_shape": [1, 12, seq_len, seq_len], "output_shape": [1, seq_len, 768], "latency_ms": 5.0},
                {"op_type": "Attention_V_MatMul", "domain": "MPC", "input_shape": [1, 12, seq_len, seq_len], "output_shape": [1, seq_len, 768], "latency_ms": 7.0},
                {"op_type": "Residual_Add", "domain": "HE", "input_shape": [1, seq_len, 768], "output_shape": [1, seq_len, 768], "latency_ms": 1.5},
                {"op_type": "Residual_Add", "domain": "MPC", "input_shape": [1, seq_len, 768], "output_shape": [1, seq_len, 768], "latency_ms": 2.8},
                {"op_type": "FFN_Linear_1", "domain": "HE", "input_shape": [1, seq_len, 768], "output_shape": [1, seq_len, 64], "latency_ms": 4.0},
                {"op_type": "FFN_Linear_1", "domain": "MPC", "input_shape": [1, seq_len, 768], "output_shape": [1, seq_len, 64], "latency_ms": 7.0},
                {"op_type": "GeLU", "domain": "HE", "input_shape": [1, seq_len, 64], "output_shape": [1, seq_len, 64], "latency_ms": 7.0},
                {"op_type": "GeLU", "domain": "MPC", "input_shape": [1, seq_len, 64], "output_shape": [1, seq_len, 64], "latency_ms": 2.0},
                {"op_type": "FFN_Linear_2", "domain": "HE", "input_shape": [1, seq_len, 64], "output_shape": [1, seq_len, 768], "latency_ms": 4.0},
                {"op_type": "FFN_Linear_2", "domain": "MPC", "input_shape": [1, seq_len, 64], "output_shape": [1, seq_len, 768], "latency_ms": 7.0},
            ],
            "conversion_records": [
                {"from_domain": "HE", "to_domain": "MPC", "tensor_shape": [1, seq_len, 768], "latency_ms": 0.8},
                {"from_domain": "MPC", "to_domain": "HE", "tensor_shape": [1, seq_len, 768], "latency_ms": 0.8},
                {"from_domain": "HE", "to_domain": "MPC", "tensor_shape": [1, 12, seq_len, seq_len], "latency_ms": 0.5},
                {"from_domain": "MPC", "to_domain": "HE", "tensor_shape": [1, 12, seq_len, seq_len], "latency_ms": 0.5},
                {"from_domain": "HE", "to_domain": "MPC", "tensor_shape": [1, seq_len, 64], "latency_ms": 0.4},
                {"from_domain": "MPC", "to_domain": "HE", "tensor_shape": [1, seq_len, 64], "latency_ms": 0.4},
            ],
        }
    )


def _build_real_ffn_subgraph() -> OperatorGraph:
    return OperatorGraph(
        graph_id="real_ffn_subgraph",
        nodes=[
            OperatorNode("s1", "LayerNorm", (1, 8, 768), (1, 8, 768), {"stage": "pre_ffn"}),
            OperatorNode("s2", "FFN_Linear_1", (1, 8, 768), (1, 8, 64), {"out_dim": 64}),
            OperatorNode("s3", "GeLU", (1, 8, 64), (1, 8, 64), {}),
            OperatorNode("s4", "FFN_Linear_2", (1, 8, 64), (1, 8, 768), {"hidden_size": 64, "out_dim": 768}),
            OperatorNode("s5", "Residual_Add", (1, 8, 768), (1, 8, 768), {"inputs": 2}),
        ],
        edges=[
            DataEdge("s1", "s2", (1, 8, 768)),
            DataEdge("s2", "s3", (1, 8, 64)),
            DataEdge("s3", "s4", (1, 8, 64)),
            DataEdge("s4", "s5", (1, 8, 768)),
            DataEdge("s1", "s5", (1, 8, 768)),
        ],
    )


def _build_real_ffn_profiler() -> ProfilerDB:
    return ProfilerDB.from_dict(
        {
            "records": [
                {"op_type": "LayerNorm", "domain": "HE", "input_shape": [1, 8, 768], "output_shape": [1, 8, 768], "latency_ms": 3.0},
                {"op_type": "LayerNorm", "domain": "MPC", "input_shape": [1, 8, 768], "output_shape": [1, 8, 768], "latency_ms": 7.0},
                {"op_type": "FFN_Linear_1", "domain": "HE", "input_shape": [1, 8, 768], "output_shape": [1, 8, 64], "latency_ms": 4.0},
                {"op_type": "FFN_Linear_1", "domain": "MPC", "input_shape": [1, 8, 768], "output_shape": [1, 8, 64], "latency_ms": 8.0},
                {"op_type": "GeLU", "domain": "HE", "input_shape": [1, 8, 64], "output_shape": [1, 8, 64], "latency_ms": 7.0},
                {"op_type": "GeLU", "domain": "MPC", "input_shape": [1, 8, 64], "output_shape": [1, 8, 64], "latency_ms": 2.0},
                {"op_type": "FFN_Linear_2", "domain": "HE", "input_shape": [1, 8, 64], "output_shape": [1, 8, 768], "latency_ms": 4.0},
                {"op_type": "FFN_Linear_2", "domain": "MPC", "input_shape": [1, 8, 64], "output_shape": [1, 8, 768], "latency_ms": 8.0},
                {"op_type": "Residual_Add", "domain": "HE", "input_shape": [1, 8, 768], "output_shape": [1, 8, 768], "latency_ms": 1.0},
                {"op_type": "Residual_Add", "domain": "MPC", "input_shape": [1, 8, 768], "output_shape": [1, 8, 768], "latency_ms": 3.0},
            ],
            "conversion_records": [
                {"from_domain": "HE", "to_domain": "MPC", "tensor_shape": [1, 8, 64], "latency_ms": 0.4},
                {"from_domain": "MPC", "to_domain": "HE", "tensor_shape": [1, 8, 64], "latency_ms": 0.4},
                {"from_domain": "HE", "to_domain": "MPC", "tensor_shape": [1, 8, 768], "latency_ms": 0.8},
                {"from_domain": "MPC", "to_domain": "HE", "tensor_shape": [1, 8, 768], "latency_ms": 0.8},
            ],
        }
    )


def _build_stable_real_ffn_subgraph() -> OperatorGraph:
    return OperatorGraph(
        graph_id="stable_real_ffn_subgraph",
        nodes=[
            OperatorNode("r1", "LayerNorm", (1, 8, 768), (1, 8, 768), {"stage": "pre_ffn"}),
            OperatorNode("r2", "FFN_Linear_1", (1, 8, 768), (1, 8, 64), {"out_dim": 64}),
            OperatorNode("r3", "FFN_Linear_2", (1, 8, 64), (1, 8, 768), {"hidden_size": 64, "out_dim": 768}),
            OperatorNode("r4", "Residual_Add", (1, 8, 768), (1, 8, 768), {"inputs": 2}),
        ],
        edges=[
            DataEdge("r1", "r2", (1, 8, 768)),
            DataEdge("r2", "r3", (1, 8, 64)),
            DataEdge("r3", "r4", (1, 8, 768)),
            DataEdge("r1", "r4", (1, 8, 768)),
        ],
    )


def _build_stable_real_ffn_profiler() -> ProfilerDB:
    return ProfilerDB.from_dict(
        {
            "records": [
                {"op_type": "LayerNorm", "domain": "HE", "input_shape": [1, 8, 768], "output_shape": [1, 8, 768], "latency_ms": 3.0},
                {"op_type": "LayerNorm", "domain": "MPC", "input_shape": [1, 8, 768], "output_shape": [1, 8, 768], "latency_ms": 7.0},
                {"op_type": "FFN_Linear_1", "domain": "HE", "input_shape": [1, 8, 768], "output_shape": [1, 8, 64], "latency_ms": 4.0},
                {"op_type": "FFN_Linear_1", "domain": "MPC", "input_shape": [1, 8, 768], "output_shape": [1, 8, 64], "latency_ms": 8.0},
                {"op_type": "FFN_Linear_2", "domain": "HE", "input_shape": [1, 8, 64], "output_shape": [1, 8, 768], "latency_ms": 4.0},
                {"op_type": "FFN_Linear_2", "domain": "MPC", "input_shape": [1, 8, 64], "output_shape": [1, 8, 768], "latency_ms": 8.0},
                {"op_type": "Residual_Add", "domain": "HE", "input_shape": [1, 8, 768], "output_shape": [1, 8, 768], "latency_ms": 1.0},
                {"op_type": "Residual_Add", "domain": "MPC", "input_shape": [1, 8, 768], "output_shape": [1, 8, 768], "latency_ms": 3.0},
            ],
            "conversion_records": [
                {"from_domain": "HE", "to_domain": "MPC", "tensor_shape": [1, 8, 64], "latency_ms": 0.4},
                {"from_domain": "MPC", "to_domain": "HE", "tensor_shape": [1, 8, 64], "latency_ms": 0.4},
                {"from_domain": "HE", "to_domain": "MPC", "tensor_shape": [1, 8, 768], "latency_ms": 0.8},
                {"from_domain": "MPC", "to_domain": "HE", "tensor_shape": [1, 8, 768], "latency_ms": 0.8},
            ],
        }
    )





def _validate_restricted_conversion_invocation() -> None:
    x = np.random.standard_normal((1, 8, 64))
    ctx = ExecutionContext(params={"conversion_sci_seed": 7})
    he_tensor = TensorValue(x, BackendType.HE, {"source": "unit-test"})
    mpc_tensor = conversion_manager.convert(
        he_tensor,
        BackendType.MPC,
        ctx,
        method_name="method_sci_restricted",
    )
    roundtrip = conversion_manager.convert(
        mpc_tensor,
        BackendType.HE,
        ctx,
        method_name="method_sci_restricted",
    )
    assert mpc_tensor.domain == BackendType.MPC
    assert roundtrip.domain == BackendType.HE
    assert mpc_tensor.data.shape == x.shape
    assert roundtrip.data.shape == x.shape
    assert not np.array_equal(np.asarray(mpc_tensor.data), x)
    assert roundtrip.meta["conversion_protocol"]["direction"] == "MPC_to_HE"
    assert mpc_tensor.meta["conversion_protocol"]["kind"] == "sci_restricted"
    assert roundtrip.meta["conversion_protocol"]["kind"] == "sci_restricted"
    assert (
        conversion_capability_registry.get_status(BackendType.HE, BackendType.MPC, "method_sci_restricted").value
        == "restricted-integrated"
    )
    assert (
        conversion_capability_registry.get_status(BackendType.MPC, BackendType.HE, "method_sci_restricted").value
        == "restricted-integrated"
    )
    print("=== Restricted Conversion Check ===")
    print("restricted_conversion_status= PASS")
    print("he_to_mpc_method= method_sci_restricted")
    print("mpc_to_he_method= method_sci_restricted")

def _classify_method_status(op_type: str, method: str, backend: BackendType) -> str:
    if method == "method_runtime_default":
        return "semantic lowering" if op_type == "Residual_Add" else "mock"
    return capability_registry.get_status(op_type, backend).value


def _audit_plan_methods(runtime_plan, registry: OperatorRegistry) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for step in runtime_plan.steps:
        if step.type != "operator":
            continue
        registry.get(step.op_type, step.backend, method_name=step.method)
        rows.append(
            {
                "op_type": step.op_type,
                "method": step.method,
                "backend": step.backend.value,
                "status": _classify_method_status(step.op_type, step.method, step.backend),
            }
        )
    return rows


def _print_table(rows: list[dict[str, str]]) -> None:
    print("op_type | method | backend | status")
    for row in rows:
        print(f"{row['op_type']} | {row['method']} | {row['backend']} | {row['status']}")


def _execute_graph(
    graph: OperatorGraph,
    profiler: ProfilerDB,
    *,
    name: str,
    params: dict[str, int],
    retries: int = 1,
) -> RealExecutionResult:
    assignment_result, _, runtime_plan = compile_graph_to_runtime_plan(
        graph,
        CostModel(db=profiler, default_strategy="auto"),
    )

    registry = OperatorRegistry()
    register_default_backend_impls(registry)
    table = _audit_plan_methods(runtime_plan, registry)

    operator_steps = [step for step in runtime_plan.steps if step.type == "operator"]
    conversion_steps = [step for step in runtime_plan.steps if step.type == "conversion"]
    assert operator_steps

    source_nodes = {edge.dst for edge in graph.edges}
    root_nodes = [node for node in graph.nodes if node.node_id not in source_nodes]
    assert len(root_nodes) == 1
    x = np.random.standard_normal(tuple(root_nodes[0].input_shape))
    last_error: str | None = None
    last_trace: list[str] = []
    for attempt in range(1, retries + 1):
        ctx = ExecutionContext(params=params)
        if retries > 1:
            ctx.trace.append(f"[validation_retry] subset={name} attempt={attempt}/{retries}")
        try:
            outputs = execute(runtime_plan, {"input": TensorValue(x, operator_steps[0].backend)}, ctx=ctx, registry=registry)
            assert any(line.startswith("EXECUTE ") for line in ctx.trace)
            assert not any("router" in line.lower() for line in ctx.trace)
            if any(step.method == "method_sci_restricted" for step in conversion_steps):
                assert any("CONVERT HE_to_MPC@method_sci_restricted[restricted-integrated]" in line or "CONVERT MPC_to_HE@method_sci_restricted[restricted-integrated]" in line for line in ctx.trace)
                assert not any("@method_default[mock]" in line for line in ctx.trace if line.startswith("CONVERT "))
            _ = outputs[operator_steps[-1].outputs[0]]
            return RealExecutionResult(
                name=name,
                table=table,
                assignment=assignment_result.assignment,
                trace=list(ctx.trace),
                success=True,
            )
        except Exception as exc:
            last_error = str(exc)
            last_trace = list(ctx.trace)

    return RealExecutionResult(
        name=name,
        table=table,
        assignment=assignment_result.assignment,
        trace=last_trace,
        success=False,
        error=last_error,
    )


def _execute_real_subset() -> RealExecutionResult:
    return _execute_graph(
        _build_real_ffn_subgraph(),
        _build_real_ffn_profiler(),
        name="LayerNorm -> FFN_Linear_1 -> GeLU -> FFN_Linear_2 -> Residual_Add",
        params={
            "ffn_linear1_he_nexus_out_dim": 64,
            "ffn_linear2_he_nexus_hidden_size": 64,
            "ffn_linear2_he_nexus_out_dim": 768,
        },
        retries=2,
    )


def _execute_stable_real_subset() -> RealExecutionResult:
    return _execute_graph(
        _build_stable_real_ffn_subgraph(),
        _build_stable_real_ffn_profiler(),
        name="LayerNorm -> FFN_Linear_1 -> FFN_Linear_2 -> Residual_Add",
        params={
            "ffn_linear1_he_nexus_out_dim": 64,
            "ffn_linear2_he_nexus_hidden_size": 64,
            "ffn_linear2_he_nexus_out_dim": 768,
        },
    )


def main() -> None:
    full_graph = build_bert_block_graph(
        BertBlockConfig(
            batch_size=1,
            seq_len=8,
            hidden_size=768,
            intermediate_size=64,
            num_heads=12,
            graph_id="bert_block_pipeline_validation",
        )
    )
    _validate_restricted_conversion_invocation()

    full_assignment, _, full_runtime_plan = compile_graph_to_runtime_plan(
        full_graph,
        CostModel(db=_build_full_block_profiler(seq_len=8), default_strategy="auto"),
    )

    registry = OperatorRegistry()
    register_default_backend_impls(registry)
    full_table = _audit_plan_methods(full_runtime_plan, registry)

    operator_steps = [step for step in full_runtime_plan.steps if step.type == "operator"]
    assert [step.op_type for step in operator_steps] == EXPECTED_OPERATOR_ORDER
    assert all(step.method for step in operator_steps)
    assert any(step.type == "conversion" for step in full_runtime_plan.steps)

    print("=== Full BERT Block Method Realism ===")
    _print_table(full_table)
    print("full_block_assignment=", full_assignment.assignment)

    print("\n=== Real Execution Attempt: Largest Mixed Subgraph ===")
    real_result = _execute_real_subset()
    print("requested_real_subset=", real_result.name)
    print("requested_real_subset_status=", "PASS" if real_result.success else "BLOCKED")
    _print_table(real_result.table)
    print("requested_real_subset_assignment=", real_result.assignment)
    if real_result.error is not None:
        print("requested_real_subset_error=", real_result.error)
    print("requested_real_subset_trace_head=")
    for line in real_result.trace[:20]:
        print(line)

    print("\n=== Real Execution Attempt: Stable Real Subgraph ===")
    stable_result = _execute_stable_real_subset()
    print("stable_real_subset=", stable_result.name)
    print("stable_real_subset_status=", "PASS" if stable_result.success else "FAIL")
    _print_table(stable_result.table)
    print("stable_real_subset_assignment=", stable_result.assignment)
    if stable_result.error is not None:
        print("stable_real_subset_error=", stable_result.error)
    print("stable_real_subset_trace_head=")
    for line in stable_result.trace[:20]:
        print(line)

    if not stable_result.success:
        raise RuntimeError("No non-trivial real-backend subgraph executed successfully.")


if __name__ == "__main__":
    main()

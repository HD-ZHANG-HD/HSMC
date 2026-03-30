from __future__ import annotations

import numpy as np

from compiler.min_cut import compile_graph_to_runtime_plan
from compiler.min_cut.cost_model import CostModel
from compiler.min_cut.profiler_db import ProfilerDB
from ir import BertBlockConfig, build_bert_block_graph
from runtime import BackendType, ExecutionContext, OperatorRegistry, TensorValue, execute


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


def _build_profiler_db(seq_len: int) -> ProfilerDB:
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


def _register_mock_impls(registry: OperatorRegistry, graph) -> None:
    node_by_op = {node.op_type: node for node in graph.nodes}

    def make_impl(op_type: str, backend: BackendType, method_name: str):
        output_shape = tuple(node_by_op[op_type].output_shape)

        def fn(inputs, ctx: ExecutionContext):
            ctx.trace.append(f"{op_type}@{backend.value}/{method_name}")
            if op_type == "Residual_Add":
                data = np.asarray(inputs[0].data, dtype=np.float64) + np.asarray(inputs[1].data, dtype=np.float64)
            elif op_type == "Attention_V_MatMul":
                left = np.asarray(inputs[0].data, dtype=np.float64)
                right = np.asarray(inputs[1].data, dtype=np.float64)
                data = np.full(output_shape, float(left.mean() + right.mean()))
            else:
                base = float(np.asarray(inputs[0].data, dtype=np.float64).mean()) if inputs else 0.0
                data = np.full(output_shape, base + float(len(op_type)))
            return TensorValue(data, backend, {"shape": list(output_shape), "method": method_name})

        return fn

    method_map = {
        "LayerNorm": {"HE": "method_he_nexus", "MPC": "method_mpc_bolt"},
        "Attention_QK_MatMul": {"HE": "method_he_nexus", "MPC": "method_mpc"},
        "Softmax": {"HE": "method_he_nexus", "MPC": "method_mpc_bolt"},
        "Attention_V_MatMul": {"HE": "method_he_nexus", "MPC": "method_mpc"},
        "Residual_Add": {"HE": "method_runtime_default", "MPC": "method_runtime_default"},
        "FFN_Linear_1": {"HE": "method_he_nexus", "MPC": "method_mpc_bolt"},
        "GeLU": {"HE": "method_he_nexus", "MPC": "method_mpc_bolt"},
        "FFN_Linear_2": {"HE": "method_he_nexus", "MPC": "method_mpc_bolt"},
    }

    for node in graph.nodes:
        for domain_name, method_name in method_map[node.op_type].items():
            backend = BackendType(domain_name)
            registry.register(node.op_type, backend, make_impl(node.op_type, backend, method_name), method_name=method_name)


def main() -> None:
    graph = build_bert_block_graph(
        BertBlockConfig(
            batch_size=1,
            seq_len=8,
            hidden_size=768,
            intermediate_size=64,
            num_heads=12,
            graph_id="bert_block_pipeline_validation",
        )
    )
    cost_model = CostModel(db=_build_profiler_db(seq_len=8), default_strategy="auto")
    assignment_result, compiler_plan, runtime_plan = compile_graph_to_runtime_plan(graph, cost_model)

    operator_steps = [step for step in runtime_plan.steps if step.type == "operator"]
    conversion_steps = [step for step in runtime_plan.steps if step.type == "conversion"]

    assert [step.op_type for step in operator_steps] == EXPECTED_OPERATOR_ORDER
    assert all(step.method for step in operator_steps)
    assert conversion_steps
    assert compiler_plan["steps"]

    registry = OperatorRegistry()
    _register_mock_impls(registry, graph)

    input_backend = operator_steps[0].backend
    x = np.random.standard_normal((1, 8, 768))
    ctx = ExecutionContext()
    outputs = execute(runtime_plan, {"input": TensorValue(x, input_backend)}, ctx=ctx, registry=registry)

    assert any(line.startswith("EXECUTE ") for line in ctx.trace)
    assert any(line.startswith("CONVERT ") for line in ctx.trace)
    assert not any("router" in line.lower() for line in ctx.trace)

    final_tensor = outputs[operator_steps[-1].outputs[0]]
    assert np.asarray(final_tensor.data).shape == tuple(graph.nodes[-1].output_shape)

    print("PASS: BERT block pipeline is plan-driven end-to-end")
    print("operator_order=", [step.op_type for step in operator_steps])
    print("methods=", [step.method for step in operator_steps])
    print("conversions=", len(conversion_steps))
    print("assignment=", assignment_result.assignment)


if __name__ == "__main__":
    main()

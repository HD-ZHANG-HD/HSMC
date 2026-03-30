from __future__ import annotations

import numpy as np

from compiler.min_cut import compile_graph_to_runtime_plan, compiler_plan_to_runtime_plan
from compiler.min_cut.cost_model import CostModel
from compiler.min_cut.domain_assignment import DataEdge, OperatorGraph, OperatorNode
from compiler.min_cut.profiler_db import ProfilerDB
from runtime import BackendType, ExecutionContext, OperatorRegistry, TensorValue, execute


def _build_contract_legal_graph() -> OperatorGraph:
    return OperatorGraph(
        graph_id="method_assignment_demo",
        nodes=[
            OperatorNode("m1", "LayerNorm", (1, 8, 768), (1, 8, 768)),
            OperatorNode("m2", "FFN_Linear_1", (1, 8, 768), (1, 8, 64)),
            OperatorNode("m3", "GeLU", (1, 8, 64), (1, 8, 64)),
            OperatorNode("m4", "FFN_Linear_2", (1, 8, 64), (1, 8, 768)),
        ],
        edges=[
            DataEdge("m1", "m2", (1, 8, 768)),
            DataEdge("m2", "m3", (1, 8, 64)),
            DataEdge("m3", "m4", (1, 8, 64)),
        ],
    )


def _build_profiler_db() -> ProfilerDB:
    return ProfilerDB.from_dict(
        {
            "records": [
                {"op_type": "LayerNorm", "domain": "HE", "input_shape": [1, 8, 768], "output_shape": [1, 8, 768], "latency_ms": 3.0},
                {"op_type": "LayerNorm", "domain": "MPC", "input_shape": [1, 8, 768], "output_shape": [1, 8, 768], "latency_ms": 6.0},
                {"op_type": "FFN_Linear_1", "domain": "HE", "input_shape": [1, 8, 768], "output_shape": [1, 8, 64], "latency_ms": 5.0},
                {"op_type": "FFN_Linear_1", "domain": "MPC", "input_shape": [1, 8, 768], "output_shape": [1, 8, 64], "latency_ms": 9.0},
                {"op_type": "GeLU", "domain": "HE", "input_shape": [1, 8, 64], "output_shape": [1, 8, 64], "latency_ms": 6.0},
                {"op_type": "GeLU", "domain": "MPC", "input_shape": [1, 8, 64], "output_shape": [1, 8, 64], "latency_ms": 2.0},
                {"op_type": "FFN_Linear_2", "domain": "HE", "input_shape": [1, 8, 64], "output_shape": [1, 8, 768], "latency_ms": 4.0},
                {"op_type": "FFN_Linear_2", "domain": "MPC", "input_shape": [1, 8, 64], "output_shape": [1, 8, 768], "latency_ms": 8.0},
            ],
            "conversion_records": [
                {"from_domain": "HE", "to_domain": "MPC", "tensor_shape": [1, 8, 64], "latency_ms": 0.5},
                {"from_domain": "MPC", "to_domain": "HE", "tensor_shape": [1, 8, 64], "latency_ms": 0.5},
                {"from_domain": "HE", "to_domain": "MPC", "tensor_shape": [1, 8, 768], "latency_ms": 0.9},
                {"from_domain": "MPC", "to_domain": "HE", "tensor_shape": [1, 8, 768], "latency_ms": 0.9},
            ],
        }
    )


def _register_mock_impls(registry: OperatorRegistry, graph: OperatorGraph) -> None:
    node_by_op = {node.op_type: node for node in graph.nodes}

    def make_impl(op_type: str, backend: BackendType, method_name: str):
        output_shape = tuple(node_by_op[op_type].output_shape)

        def fn(inputs, ctx: ExecutionContext):
            ctx.trace.append(f"{op_type}@{backend.value}/{method_name}")
            base = float(np.asarray(inputs[0].data).mean()) if inputs else 0.0
            data = np.full(output_shape, base + float(len(op_type)))
            return TensorValue(data, backend, {"shape": list(output_shape), "method": method_name})

        return fn

    method_map = {
        "LayerNorm": {"HE": "method_he_nexus", "MPC": "method_mpc_bolt"},
        "FFN_Linear_1": {"HE": "method_he_nexus", "MPC": "method_mpc_bolt"},
        "GeLU": {"HE": "method_he_nexus", "MPC": "method_mpc_bolt"},
        "FFN_Linear_2": {"HE": "method_he_nexus", "MPC": "method_mpc_bolt"},
    }

    for node in graph.nodes:
        for domain_name, method_name in method_map[node.op_type].items():
            backend = BackendType(domain_name)
            registry.register(
                node.op_type,
                backend,
                make_impl(node.op_type, backend, method_name),
                method_name=method_name,
            )


def _validate_invalid_contract_rejection() -> None:
    graph = OperatorGraph(
        graph_id="invalid_layernorm_contract",
        nodes=[OperatorNode("bad1", "LayerNorm", (2, 16, 768), (2, 16, 768))],
        edges=[],
    )
    compiler_plan = {
        "graph_id": graph.graph_id,
        "assignment": {"bad1": "HE"},
        "steps": [
            {
                "step_id": "s1",
                "kind": "operator",
                "node_id": "bad1",
                "op_type": "LayerNorm",
                "domain": "HE",
                "input_shape": [2, 16, 768],
                "output_shape": [2, 16, 768],
                "estimated_latency_ms": 1.0,
            }
        ],
    }
    try:
        compiler_plan_to_runtime_plan(graph, compiler_plan)
    except ValueError as exc:
        assert "contract validation" in str(exc)
    else:
        raise AssertionError("Expected invalid HE LayerNorm contract to fail explicitly")


def main() -> None:
    graph = _build_contract_legal_graph()
    db = _build_profiler_db()
    cost_model = CostModel(db=db, default_strategy="auto")

    assignment_result, compiler_plan, runtime_plan = compile_graph_to_runtime_plan(graph, cost_model)

    operator_steps = [step for step in runtime_plan.steps if step.type == "operator"]
    assert operator_steps, "Expected operator steps in runtime plan"
    assert all(step.method for step in operator_steps), "Every operator step must carry an explicit method"
    assert any(step.method == "method_he_nexus" for step in operator_steps)
    assert any(step.method == "method_mpc_bolt" for step in operator_steps)
    assert any(step.type == "conversion" for step in runtime_plan.steps), "Expected explicit conversion steps"

    registry = OperatorRegistry()
    _register_mock_impls(registry, graph)

    input_backend = operator_steps[0].backend
    x = np.arange(8 * 768, dtype=np.float64).reshape(1, 8, 768)
    ctx = ExecutionContext()
    outputs = execute(runtime_plan, {"input": TensorValue(x, input_backend)}, ctx=ctx, registry=registry)

    assert compiler_plan["steps"]
    assert outputs["m4__out"].domain == operator_steps[-1].backend
    assert any(line.startswith("EXECUTE ") for line in ctx.trace)
    assert any(line.startswith("CONVERT ") for line in ctx.trace)

    _validate_invalid_contract_rejection()

    print("PASS: compiler assigns explicit methods per operator step and validates contracts")
    print(f"assignment={assignment_result.assignment}")
    print("methods=", [step.method for step in operator_steps])


if __name__ == "__main__":
    main()

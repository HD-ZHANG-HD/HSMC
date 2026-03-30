
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from compiler.min_cut import compile_graph_to_runtime_plan
from compiler.min_cut.cost_model import CostModel
from compiler.min_cut.profiler_db import ProfilerDB
from compiler.min_cut.runtime_plan_adapter import compiler_plan_to_runtime_plan
from compiler.min_cut.plan_builder import build_execution_plan
from ir import DataEdge, OperatorGraph, OperatorNode
from runtime import (
    BackendType,
    ExecutionContext,
    NetworkConfig,
    NetworkModel,
    OperatorRegistry,
    ProfilingCollector,
    TensorValue,
    execute,
)


@dataclass(frozen=True)
class SimOpSpec:
    output_shape: tuple[int, ...]
    compute_ms: float
    scale: float = 1.0


CALIBRATION_SPECS: dict[tuple[str, str], SimOpSpec] = {
    ("LayerNorm", "HE"): SimOpSpec((1, 8, 768), 3.8, 0.99),
    ("LayerNorm", "MPC"): SimOpSpec((1, 8, 768), 4.5, 1.0),
    ("FFN_Linear_1", "HE"): SimOpSpec((1, 8, 64), 4.6, 0.99),
    ("FFN_Linear_1", "MPC"): SimOpSpec((1, 8, 64), 5.8, 1.0),
    ("GeLU", "HE"): SimOpSpec((1, 8, 64), 7.5, 0.99),
    ("GeLU", "MPC"): SimOpSpec((1, 8, 64), 1.1, 1.0),
    ("FFN_Linear_2", "HE"): SimOpSpec((1, 8, 768), 4.4, 0.99),
    ("FFN_Linear_2", "MPC"): SimOpSpec((1, 8, 768), 5.5, 1.0),
    ("Residual_Add", "HE"): SimOpSpec((1, 8, 768), 0.8, 0.99),
    ("Residual_Add", "MPC"): SimOpSpec((1, 8, 768), 0.6, 1.0),
}

OPERATOR_METHODS: dict[tuple[str, str], str] = {
    ("LayerNorm", "HE"): "method_he_nexus",
    ("LayerNorm", "MPC"): "method_mpc_bolt",
    ("FFN_Linear_1", "HE"): "method_he_nexus",
    ("FFN_Linear_1", "MPC"): "method_mpc_bolt",
    ("GeLU", "HE"): "method_he_nexus",
    ("GeLU", "MPC"): "method_mpc_bolt",
    ("FFN_Linear_2", "HE"): "method_he_nexus",
    ("FFN_Linear_2", "MPC"): "method_mpc_bolt",
    ("Residual_Add", "HE"): "method_runtime_default",
    ("Residual_Add", "MPC"): "method_runtime_default",
}

OPERATOR_COMM: dict[str, dict[str, int]] = {
    "LayerNorm|MPC": {"comm_bytes": 48_000, "comm_rounds": 3},
    "FFN_Linear_1|MPC": {"comm_bytes": 40_000, "comm_rounds": 3},
    "GeLU|MPC": {"comm_bytes": 18_000, "comm_rounds": 2},
    "FFN_Linear_2|MPC": {"comm_bytes": 40_000, "comm_rounds": 3},
    "Residual_Add|MPC": {"comm_bytes": 8_000, "comm_rounds": 1},
}

CONVERSION_COMM: dict[str, dict[str, int]] = {
    "HE_to_MPC|method_sci_restricted|ffn": {"comm_bytes": 64_000, "comm_rounds": 2},
    "MPC_to_HE|method_sci_restricted|ffn": {"comm_bytes": 64_000, "comm_rounds": 2},
}


def _sleep_ms(duration_ms: float) -> None:
    time.sleep(max(duration_ms, 0.0) / 1000.0)


def build_ffn_graph() -> OperatorGraph:
    return OperatorGraph(
        graph_id="profiling_ffn_graph",
        nodes=[
            OperatorNode("n1", "LayerNorm", (1, 8, 768), (1, 8, 768), {"stage": "pre_ffn"}),
            OperatorNode("n2", "FFN_Linear_1", (1, 8, 768), (1, 8, 64), {"out_dim": 64}),
            OperatorNode("n3", "GeLU", (1, 8, 64), (1, 8, 64), {}),
            OperatorNode("n4", "FFN_Linear_2", (1, 8, 64), (1, 8, 768), {"hidden_size": 64, "out_dim": 768}),
            OperatorNode("n5", "Residual_Add", (1, 8, 768), (1, 8, 768), {"inputs": 2}),
        ],
        edges=[
            DataEdge("n1", "n2", (1, 8, 768)),
            DataEdge("n2", "n3", (1, 8, 64)),
            DataEdge("n3", "n4", (1, 8, 64)),
            DataEdge("n4", "n5", (1, 8, 768)),
            DataEdge("n1", "n5", (1, 8, 768)),
        ],
    )


def _residual_add(inputs: list[np.ndarray]) -> np.ndarray:
    return np.asarray(inputs[0], dtype=np.float64) + np.asarray(inputs[1], dtype=np.float64)


def build_mock_registry() -> OperatorRegistry:
    registry = OperatorRegistry()
    for (op_type, backend_name), spec in CALIBRATION_SPECS.items():
        backend = BackendType(backend_name)
        method = OPERATOR_METHODS[(op_type, backend_name)]

        def make_fn(op_type=op_type, backend=backend, spec=spec):
            def fn(inputs, ctx):
                _sleep_ms(spec.compute_ms)
                arrays = [np.asarray(t.data, dtype=np.float64) for t in inputs]
                if op_type == "Residual_Add":
                    out = _residual_add(arrays)
                elif op_type in {"LayerNorm", "GeLU"}:
                    out = arrays[0] * spec.scale
                elif op_type == "FFN_Linear_1":
                    out = np.full(spec.output_shape, float(np.mean(arrays[0])) * spec.scale, dtype=np.float64)
                elif op_type == "FFN_Linear_2":
                    out = np.full(spec.output_shape, float(np.mean(arrays[0])) * spec.scale, dtype=np.float64)
                else:
                    out = np.full(spec.output_shape, float(np.mean(arrays[0])) * spec.scale, dtype=np.float64)
                return TensorValue(out, backend)
            return fn

        fn = make_fn()
        registry.register(op_type, backend, fn)
        registry.register(op_type, backend, fn, method_name=method)
    return registry


def _uniform_assignment(graph: OperatorGraph, domain: str) -> dict[str, str]:
    return {node.node_id: domain for node in graph.nodes}


def _mixed_assignment() -> dict[str, str]:
    return {
        "n1": "HE",
        "n2": "HE",
        "n3": "MPC",
        "n4": "HE",
        "n5": "HE",
    }


def _profile_plan(graph: OperatorGraph, runtime_plan, network_config: NetworkConfig, registry: OperatorRegistry) -> ProfilingCollector:
    collector = ProfilingCollector()
    ctx = ExecutionContext(
        params={
            "profiling_operator_comm": OPERATOR_COMM,
            "profiling_conversion_comm": CONVERSION_COMM,
            "conversion_sci_seed": 11,
        },
        profiling_collector=collector,
        network_config=network_config,
    )
    root_nodes = {edge.dst for edge in graph.edges}
    source = next(node for node in graph.nodes if node.node_id not in root_nodes)
    x = np.random.standard_normal(tuple(source.input_shape))
    operator_steps = [step for step in runtime_plan.steps if step.type == "operator"]
    assert operator_steps
    execute(runtime_plan, {"input": TensorValue(x, operator_steps[0].backend)}, ctx=ctx, registry=registry)
    return collector


def _baseline_conversion_records(network_config: NetworkConfig) -> list[dict]:
    specs = [
        ("HE", "MPC", "method_default", "generic", (1, 8, 768), 128_000, 2),
        ("MPC", "HE", "method_default", "generic", (1, 8, 768), 128_000, 2),
        ("HE", "MPC", "method_default", "generic", (1, 8, 64), 16_000, 2),
        ("MPC", "HE", "method_default", "generic", (1, 8, 64), 16_000, 2),
        ("HE", "MPC", "method_sci_restricted", "ffn", (1, 8, 64), 64_000, 2),
        ("MPC", "HE", "method_sci_restricted", "ffn", (1, 8, 64), 64_000, 2),
    ]
    records = []
    for from_domain, to_domain, method, layout_family, shape, comm_bytes, comm_rounds in specs:
        local_ms = 0.25
        records.append({
            "from_domain": from_domain,
            "to_domain": to_domain,
            "method": method,
            "layout_family": layout_family,
            "tensor_shape": list(shape),
            "local_compute_ms": local_ms,
            "comm_bytes": comm_bytes,
            "comm_rounds": comm_rounds,
            "total_latency_ms": NetworkModel.estimate_latency(local_ms, comm_bytes, comm_rounds, network_config),
        })
    return records


def _build_profile_db_for_config(graph: OperatorGraph, network_config: NetworkConfig, registry: OperatorRegistry) -> ProfilerDB:
    payload_records: list[dict] = []
    payload_conversions: list[dict] = []
    for assignment in (_uniform_assignment(graph, "HE"), _uniform_assignment(graph, "MPC"), _mixed_assignment()):
        compiler_plan = build_execution_plan(
            graph,
            assignment,
            CostModel(
                ProfilerDB.from_dict(
                    {
                        "records": [
                            {
                                "op_type": node.op_type,
                                "domain": domain,
                                "method": OPERATOR_METHODS[(node.op_type, domain)],
                                "input_shape": list(node.input_shape),
                                "output_shape": list(node.output_shape),
                                "latency_ms": CALIBRATION_SPECS[(node.op_type, domain)].compute_ms,
                            }
                            for node in graph.nodes
                            for domain in ("HE", "MPC")
                        ],
                        "conversion_records": [
                            {
                                "from_domain": "HE",
                                "to_domain": "MPC",
                                "method": "method_sci_restricted",
                                "layout_family": "ffn",
                                "tensor_shape": [1, 8, 64],
                                "latency_ms": 1.0,
                            },
                            {
                                "from_domain": "MPC",
                                "to_domain": "HE",
                                "method": "method_sci_restricted",
                                "layout_family": "ffn",
                                "tensor_shape": [1, 8, 64],
                                "latency_ms": 1.0,
                            },
                            {
                                "from_domain": "HE",
                                "to_domain": "MPC",
                                "method": "method_default",
                                "layout_family": "generic",
                                "tensor_shape": [1, 8, 768],
                                "latency_ms": 1.0,
                            },
                            {
                                "from_domain": "MPC",
                                "to_domain": "HE",
                                "method": "method_default",
                                "layout_family": "generic",
                                "tensor_shape": [1, 8, 768],
                                "latency_ms": 1.0,
                            },
                        ],
                    }
                ),
                default_strategy="exact",
            ),
            include_baselines=False,
        )
        runtime_plan = compiler_plan_to_runtime_plan(graph, compiler_plan)
        collector = _profile_plan(graph, runtime_plan, network_config, registry)
        payload = collector.export_payload()
        payload_records.extend(payload["records"])
        payload_conversions.extend(payload["conversion_records"])
    payload_conversions.extend(_baseline_conversion_records(network_config))
    return ProfilerDB.from_dict({
        "schema_version": "2.0",
        "domains": ["HE", "MPC"],
        "records": payload_records,
        "conversion_records": payload_conversions,
    })


def _estimate_total_latency(graph: OperatorGraph, assignment: dict[str, str], cost_model: CostModel) -> float:
    node_map = {node.node_id: node for node in graph.nodes}
    total = 0.0
    for edge in graph.edges:
        src_domain = assignment[edge.src]
        dst_domain = assignment[edge.dst]
        if src_domain != dst_domain:
            src_op = node_map[edge.src].op_type
            dst_op = node_map[edge.dst].op_type
            method = "method_sci_restricted" if (src_op, dst_op) in {("FFN_Linear_1", "GeLU"), ("GeLU", "FFN_Linear_2")} else "method_default"
            layout = "ffn" if method == "method_sci_restricted" else "generic"
            total += cost_model.estimate_conversion_cost(
                edge.tensor_shape,
                src_domain,  # type: ignore[arg-type]
                dst_domain,  # type: ignore[arg-type]
                method=method,
                layout_family=layout,
            ).latency_ms
    for node in graph.nodes:
        total += cost_model.estimate_node_cost(
            node.op_type,
            assignment[node.node_id],  # type: ignore[arg-type]
            node.input_shape,
            node.output_shape,
        ).latency_ms
    return total


def _print_profile_table(collector: ProfilingCollector) -> None:
    print("operator_metrics:")
    for rec in collector.operator_records:
        print(
            f"  {rec.op_type} | {rec.method} | {rec.backend} | "
            f"local_ms={rec.local_compute_ms:.3f} comm_bytes={rec.comm_bytes} "
            f"comm_rounds={rec.comm_rounds} total_ms={rec.total_latency_ms:.3f}"
        )
    print("conversion_metrics:")
    for rec in collector.conversion_records:
        print(
            f"  {rec.direction} | {rec.method} | {rec.layout_family} | "
            f"local_ms={rec.local_compute_ms:.3f} comm_bytes={rec.comm_bytes} "
            f"comm_rounds={rec.comm_rounds} total_ms={rec.total_latency_ms:.3f}"
        )


def main() -> None:
    graph = build_ffn_graph()
    registry = build_mock_registry()
    # 10 Mps
    # configs = {
    #     "A": NetworkConfig(bandwidth_bytes_per_sec=1_250_000.0, rtt_ms=1.0),
    #     "B": NetworkConfig(bandwidth_bytes_per_sec=1_250_000.0, rtt_ms=20.0),
    #     "C": NetworkConfig(bandwidth_bytes_per_sec=1_250_000.0, rtt_ms=50.0),
    # }
    # 100 Mps
    configs = {
        "A": NetworkConfig(bandwidth_bytes_per_sec=12_500_000.0, rtt_ms=1.0),
        "B": NetworkConfig(bandwidth_bytes_per_sec=12_500_000.0, rtt_ms=20.0),
        "C": NetworkConfig(bandwidth_bytes_per_sec=12_500_000.0, rtt_ms=50.0),
    }
    # 200 Mps
    # configs = {
    #     "A": NetworkConfig(bandwidth_bytes_per_sec=25_000_000.0, rtt_ms=1.0),
    #     "B": NetworkConfig(bandwidth_bytes_per_sec=25_000_000.0, rtt_ms=20.0),
    #     "C": NetworkConfig(bandwidth_bytes_per_sec=25_000_000.0, rtt_ms=50.0),
    # }
    # 1 Gps
    # configs = {
    #     "A": NetworkConfig(bandwidth_bytes_per_sec=125_000_000.0, rtt_ms=1.0),
    #     "B": NetworkConfig(bandwidth_bytes_per_sec=125_000_000.0, rtt_ms=20.0),
    #     "C": NetworkConfig(bandwidth_bytes_per_sec=125_000_000.0, rtt_ms=50.0),
    # }
    # 3Gps
    # configs = {
    #     "A": NetworkConfig(bandwidth_bytes_per_sec=375_000_000.0, rtt_ms=0.5),
    #     "B": NetworkConfig(bandwidth_bytes_per_sec=375_000_000.0, rtt_ms=0.8),
    #     "C": NetworkConfig(bandwidth_bytes_per_sec=375_000_000.0, rtt_ms=1.0),
    # }
    summaries: dict[str, dict[str, object]] = {}

    for label, network_config in configs.items():
        db = _build_profile_db_for_config(graph, network_config, registry)
        cost_model = CostModel(db=db, default_strategy="exact")
        assignment_result, _, runtime_plan = compile_graph_to_runtime_plan(graph, cost_model)
        collector = _profile_plan(graph, runtime_plan, network_config, registry)
        estimated_total_latency = _estimate_total_latency(graph, assignment_result.assignment, cost_model)
        operator_steps = [step for step in runtime_plan.steps if step.type == "operator"]
        conversion_steps = [step for step in runtime_plan.steps if step.type == "conversion"]
        summaries[label] = {
            "assignment": dict(assignment_result.assignment),
            "methods": {step.op_type: step.method for step in operator_steps},
            "conversion_count": len(conversion_steps),
            "estimated_total_latency_ms": estimated_total_latency,
        }
        print(f"=== Config {label} ===")
        print(
            f"network=bandwidth:{network_config.bandwidth_bytes_per_sec:.0f}Bps rtt_ms:{network_config.rtt_ms:.1f}"
        )
        _print_profile_table(collector)
        print("plan_summary:")
        print(f"  assignment={assignment_result.assignment}")
        print(f"  methods={{step.op_type: step.method for step in operator_steps}} -> {summaries[label]['methods']}")
        print(f"  conversion_count={len(conversion_steps)}")
        print(f"  estimated_total_latency_ms={estimated_total_latency:.3f}")
        print()

    base_assignment = summaries["A"]["assignment"]
    policy_changed = False
    changed_nodes: dict[str, tuple[str, str]] = {}
    for label in ("B", "C"):
        for node_id, domain in summaries[label]["assignment"].items():
            if base_assignment[node_id] != domain:
                policy_changed = True
                changed_nodes[node_id] = (base_assignment[node_id], domain)
    print("=== Policy Comparison ===")
    print(f"policy_changed={policy_changed}")
    if policy_changed:
        for node_id, (old_domain, new_domain) in changed_nodes.items():
            print(f"  node={node_id} changed {old_domain} -> {new_domain}")
    else:
        print("  planner stayed stable; graph or communication pressure may still be too small.")
        print("=== Forced Sensitivity Demo ===")
        original_specs = dict(CALIBRATION_SPECS)
        original_operator_comm = dict(OPERATOR_COMM)
        original_conversion_comm = dict(CONVERSION_COMM)
        try:
            CALIBRATION_SPECS.update({
                ("GeLU", "HE"): SimOpSpec((1, 8, 64), 9.0, 0.99),
                ("GeLU", "MPC"): SimOpSpec((1, 8, 64), 0.2, 1.0),
            })
            OPERATOR_COMM.clear()
            OPERATOR_COMM.update({
                "LayerNorm|MPC": {"comm_bytes": 16_000, "comm_rounds": 2},
                "FFN_Linear_1|MPC": {"comm_bytes": 12_000, "comm_rounds": 2},
                "GeLU|MPC": {"comm_bytes": 4_000, "comm_rounds": 1},
                "FFN_Linear_2|MPC": {"comm_bytes": 12_000, "comm_rounds": 2},
                "Residual_Add|MPC": {"comm_bytes": 4_000, "comm_rounds": 1},
            })
            CONVERSION_COMM.clear()
            CONVERSION_COMM.update({
                "HE_to_MPC|method_sci_restricted|ffn": {"comm_bytes": 24_000, "comm_rounds": 1},
                "MPC_to_HE|method_sci_restricted|ffn": {"comm_bytes": 24_000, "comm_rounds": 1},
            })
            forced_registry = build_mock_registry()
            forced_summaries = {}
            for forced_label in ("A", "C"):
                forced_config = configs[forced_label]
                forced_db = _build_profile_db_for_config(graph, forced_config, forced_registry)
                forced_cost_model = CostModel(db=forced_db, default_strategy="exact")
                forced_assignment, _, _ = compile_graph_to_runtime_plan(graph, forced_cost_model)
                forced_summaries[forced_label] = dict(forced_assignment.assignment)
                print(f"  forced_config_{forced_label}={forced_assignment.assignment}")
            forced_changed = forced_summaries["A"] != forced_summaries["C"]
            print(f"  forced_policy_changed={forced_changed}")
            if forced_changed:
                for node_id, domain in forced_summaries["C"].items():
                    if forced_summaries["A"][node_id] != domain:
                        print(f"    node={node_id} changed {forced_summaries['A'][node_id]} -> {domain}")
        finally:
            CALIBRATION_SPECS.clear()
            CALIBRATION_SPECS.update(original_specs)
            OPERATOR_COMM.clear()
            OPERATOR_COMM.update(original_operator_comm)
            CONVERSION_COMM.clear()
            CONVERSION_COMM.update(original_conversion_comm)


if __name__ == "__main__":
    main()

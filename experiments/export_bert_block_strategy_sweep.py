from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from compiler.min_cut import compile_graph_to_runtime_plan
from compiler.min_cut.cost_model import CostModel
from compiler.min_cut.profiler_db import ProfilerDB
from ir import BertBlockConfig, OperatorGraph, build_bert_block_graph
from runtime import NetworkConfig, NetworkModel


OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "bert_block_strategy_sweep"


@dataclass(frozen=True)
class NetworkSweepConfig:
    name: str
    bandwidth_label: str
    bandwidth_bytes_per_sec: float
    rtt_ms: float


@dataclass(frozen=True)
class SyntheticOperatorSpec:
    he_local_ms: float
    mpc_local_ms: float
    mpc_comm_bytes: int
    mpc_comm_rounds: int
    notes: str = ""


def _bits_to_bytes_per_sec(bits_per_sec: float) -> float:
    return bits_per_sec / 8.0


def _numel(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def _shape_list(shape: Iterable[int]) -> list[int]:
    return [int(v) for v in shape]


def _slugify(label: str) -> str:
    return label.lower().replace(" ", "_").replace("=", "").replace(",", "").replace(".", "p")


def _network_configs() -> list[NetworkSweepConfig]:
    return [
        NetworkSweepConfig("10Mbps_RTT40ms", "10 Mbps", _bits_to_bytes_per_sec(10_000_000.0), 40.0),
        NetworkSweepConfig("10Mbps_RTT80ms", "10 Mbps", _bits_to_bytes_per_sec(10_000_000.0), 80.0),
        NetworkSweepConfig("100Mbps_RTT40ms", "100 Mbps", _bits_to_bytes_per_sec(100_000_000.0), 40.0),
        NetworkSweepConfig("100Mbps_RTT80ms", "100 Mbps", _bits_to_bytes_per_sec(100_000_000.0), 80.0),
        NetworkSweepConfig("200Mbps_RTT40ms", "200 Mbps", _bits_to_bytes_per_sec(200_000_000.0), 40.0),
        NetworkSweepConfig("200Mbps_RTT80ms", "200 Mbps", _bits_to_bytes_per_sec(200_000_000.0), 80.0),
        NetworkSweepConfig("1Gbps_RTT0.8ms", "1 Gbps", _bits_to_bytes_per_sec(1_000_000_000.0), 0.8),
        NetworkSweepConfig("1Gbps_RTT1ms", "1 Gbps", _bits_to_bytes_per_sec(1_000_000_000.0), 1.0),
        NetworkSweepConfig("1Gbps_RTT5ms", "1 Gbps", _bits_to_bytes_per_sec(1_000_000_000.0), 5.0),
        NetworkSweepConfig("3Gbps_RTT0.8ms", "3 Gbps", _bits_to_bytes_per_sec(3_000_000_000.0), 0.8),
        NetworkSweepConfig("3Gbps_RTT1ms", "3 Gbps", _bits_to_bytes_per_sec(3_000_000_000.0), 1.0),
        NetworkSweepConfig("3Gbps_RTT5ms", "3 Gbps", _bits_to_bytes_per_sec(3_000_000_000.0), 5.0),
    ]


OPERATOR_SPECS: dict[str, SyntheticOperatorSpec] = {
    "LayerNorm": SyntheticOperatorSpec(
        he_local_ms=4.6,
        mpc_local_ms=0.7,
        mpc_comm_bytes=36_000,
        mpc_comm_rounds=3,
        notes="HE path restricted-integrated on [B,S,768]; MPC path network-aware synthetic cost.",
    ),
    "Attention_QK_MatMul": SyntheticOperatorSpec(
        he_local_ms=1_000.0,
        mpc_local_ms=4.4,
        mpc_comm_bytes=140_000,
        mpc_comm_rounds=3,
        notes=(
            "HE entry is a synthetic high-cost placeholder only. Current BERT-block IR shape is not valid for "
            "method_he_nexus, so runtime plan export expects MPC for this node."
        ),
    ),
    "Softmax": SyntheticOperatorSpec(
        he_local_ms=12.8,
        mpc_local_ms=1.8,
        mpc_comm_bytes=90_000,
        mpc_comm_rounds=4,
        notes="Softmax participates in planner sweep with network-aware MPC cost.",
    ),
    "Attention_V_MatMul": SyntheticOperatorSpec(
        he_local_ms=8.2,
        mpc_local_ms=2.6,
        mpc_comm_bytes=70_000,
        mpc_comm_rounds=3,
        notes="HE path represents restricted NEXUS attention-V contract when chosen.",
    ),
    "Residual_Add": SyntheticOperatorSpec(
        he_local_ms=1.6,
        mpc_local_ms=0.15,
        mpc_comm_bytes=8_000,
        mpc_comm_rounds=1,
        notes="Semantic lowering; still modeled as backend-specific latency for planning.",
    ),
    "FFN_Linear_1": SyntheticOperatorSpec(
        he_local_ms=4.9,
        mpc_local_ms=3.4,
        mpc_comm_bytes=80_000,
        mpc_comm_rounds=3,
        notes="HE path mirrors restricted NEXUS FFN contract; MPC cost is network-aware synthetic.",
    ),
    "GeLU": SyntheticOperatorSpec(
        he_local_ms=7.4,
        mpc_local_ms=0.8,
        mpc_comm_bytes=18_000,
        mpc_comm_rounds=2,
        notes="Real MPC method exists; HE is approximate local wrapper.",
    ),
    "FFN_Linear_2": SyntheticOperatorSpec(
        he_local_ms=4.7,
        mpc_local_ms=3.1,
        mpc_comm_bytes=72_000,
        mpc_comm_rounds=3,
        notes="HE path mirrors restricted NEXUS FFN contract; MPC cost is network-aware synthetic.",
    ),
}


def _operator_method(op_type: str, backend: str) -> str:
    if op_type == "Attention_QK_MatMul" and backend == "MPC":
        return "method_mpc"
    if op_type == "Attention_V_MatMul" and backend == "MPC":
        return "method_mpc"
    if op_type == "Residual_Add":
        return "method_runtime_default"
    if backend == "HE":
        return "method_he_nexus"
    return "method_mpc_bolt"


def _conversion_layout_for_shape(shape: tuple[int, ...]) -> str:
    if len(shape) == 3 and shape[-1] in {768, 3072}:
        return "ffn"
    if len(shape) == 4:
        return "attention"
    return "generic"


def _conversion_bytes(shape: tuple[int, ...], layout_family: str) -> int:
    base = _numel(shape) * 16
    if layout_family == "attention":
        return max(24_000, base * 2)
    if layout_family == "ffn":
        return max(64_000, base * 2)
    return max(32_000, base)


def _conversion_rounds(layout_family: str) -> int:
    if layout_family == "attention":
        return 3
    if layout_family == "ffn":
        return 2
    return 2


def build_planning_profiler_db(graph: OperatorGraph, network: NetworkConfig) -> ProfilerDB:
    records: list[dict[str, object]] = []
    conversion_records: list[dict[str, object]] = []

    for node in graph.nodes:
        spec = OPERATOR_SPECS[node.op_type]
        he_method = _operator_method(node.op_type, "HE")
        mpc_method = _operator_method(node.op_type, "MPC")
        records.append({
            "op_type": node.op_type,
            "domain": "HE",
            "method": he_method,
            "input_shape": _shape_list(node.input_shape),
            "output_shape": _shape_list(node.output_shape),
            "local_compute_ms": spec.he_local_ms,
            "comm_bytes": 0,
            "comm_rounds": 0,
            "total_latency_ms": spec.he_local_ms,
            "metadata": {
                "synthetic_profile": True,
                "notes": spec.notes,
            },
        })
        mpc_total = NetworkModel.estimate_latency(
            spec.mpc_local_ms,
            spec.mpc_comm_bytes,
            spec.mpc_comm_rounds,
            network,
        )
        records.append({
            "op_type": node.op_type,
            "domain": "MPC",
            "method": mpc_method,
            "input_shape": _shape_list(node.input_shape),
            "output_shape": _shape_list(node.output_shape),
            "local_compute_ms": spec.mpc_local_ms,
            "comm_bytes": spec.mpc_comm_bytes,
            "comm_rounds": spec.mpc_comm_rounds,
            "total_latency_ms": mpc_total,
            "metadata": {
                "synthetic_profile": True,
                "notes": spec.notes,
            },
        })

    seen_conv: set[tuple[str, str, tuple[int, ...], str]] = set()
    for edge in graph.edges:
        layout_family = _conversion_layout_for_shape(edge.tensor_shape)
        comm_bytes = _conversion_bytes(edge.tensor_shape, layout_family)
        comm_rounds = _conversion_rounds(layout_family)
        local_ms = 0.35
        for from_domain, to_domain in (("HE", "MPC"), ("MPC", "HE")):
            key = (from_domain, to_domain, edge.tensor_shape, layout_family)
            if key in seen_conv:
                continue
            seen_conv.add(key)
            conversion_records.append({
                "from_domain": from_domain,
                "to_domain": to_domain,
                "method": "method_default",
                "layout_family": layout_family,
                "tensor_shape": _shape_list(edge.tensor_shape),
                "local_compute_ms": local_ms,
                "comm_bytes": comm_bytes,
                "comm_rounds": comm_rounds,
                "total_latency_ms": NetworkModel.estimate_latency(local_ms, comm_bytes, comm_rounds, network),
                "metadata": {
                    "synthetic_profile": True,
                    "notes": "Planning-only conversion cost. Attention uses simulated layout-aware records until full conversion support lands.",
                },
            })

    return ProfilerDB.from_dict({
        "schema_version": "2.0",
        "domains": ["HE", "MPC"],
        "records": records,
        "conversion_records": conversion_records,
    })


def _conversion_edges(graph: OperatorGraph, assignment: dict[str, str]) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for edge in graph.edges:
        edges.append({
            "src": edge.src,
            "dst": edge.dst,
            "kind": "dataflow",
            "tensor_shape": _shape_list(edge.tensor_shape),
        })
        src_domain = assignment[edge.src]
        dst_domain = assignment[edge.dst]
        if src_domain != dst_domain:
            edges.append({
                "src": edge.src,
                "dst": edge.dst,
                "kind": "conversion",
                "from_domain": src_domain,
                "to_domain": dst_domain,
                "tensor_shape": _shape_list(edge.tensor_shape),
            })
    return edges


def _dot_for_strategy(config_name: str, nodes: list[dict[str, object]], edges: list[dict[str, object]], total_ms: float) -> str:
    lines = [
        "digraph Strategy {",
        "  rankdir=LR;",
        f'  label="{config_name} | total={total_ms:.3f} ms";',
        "  labelloc=t;",
    ]
    for node in nodes:
        backend = str(node["backend"])
        color = "#d7f0d2" if backend == "HE" else "#d5e8ff"
        label = f'{node["node_id"]}\n{node["op_type"]}\n{backend}\n{node["method"]}'
        lines.append(f'  {node["node_id"]} [shape=box, style=filled, fillcolor="{color}", label="{label}"];')
    for edge in edges:
        if edge["kind"] == "dataflow":
            lines.append(f'  {edge["src"]} -> {edge["dst"]} [color="#666666"];')
        else:
            label = f'{edge["from_domain"]}->{edge["to_domain"]}'
            lines.append(f'  {edge["src"]} -> {edge["dst"]} [color="#cc0000", penwidth=2.0, label="{label}"];')
    lines.append("}")
    return "\n".join(lines)


def _policy_change_summary(results: list[dict[str, object]]) -> dict[str, object]:
    node_backends: dict[str, set[str]] = {}
    node_methods: dict[str, set[str]] = {}
    conversion_counts = {result["config_name"]: result["conversion_count"] for result in results}
    for result in results:
        for node in result["nodes"]:
            node_id = str(node["node_id"])
            node_backends.setdefault(node_id, set()).add(str(node["backend"]))
            node_methods.setdefault(node_id, set()).add(str(node["method"]))
    return {
        "backend_changes": {node_id: sorted(values) for node_id, values in node_backends.items() if len(values) > 1},
        "method_changes": {node_id: sorted(values) for node_id, values in node_methods.items() if len(values) > 1},
        "conversion_count_by_config": conversion_counts,
        "min_latency_config": min(results, key=lambda item: float(item["estimated_total_latency_ms"]))["config_name"],
        "max_latency_config": max(results, key=lambda item: float(item["estimated_total_latency_ms"]))["config_name"],
    }


def _text_summary(results: list[dict[str, object]], policy_summary: dict[str, object]) -> str:
    lines = ["BERT block strategy sweep summary:"]
    for result in results:
        lines.append(
            f'  {result["config_name"]}: total={result["estimated_total_latency_ms"]:.3f} ms | '
            f'conversions={result["conversion_count"]} | assignment='
            + ", ".join(f'{node["node_id"]}:{node["backend"]}' for node in result["nodes"])
        )
    lines.append("policy changes:")
    backend_changes = policy_summary["backend_changes"]
    method_changes = policy_summary["method_changes"]
    if backend_changes:
        for node_id, values in backend_changes.items():
            lines.append(f"  backend {node_id}: {' -> '.join(values)}")
    else:
        lines.append("  backend changes: none")
    if method_changes:
        for node_id, values in method_changes.items():
            lines.append(f"  method {node_id}: {' -> '.join(values)}")
    else:
        lines.append("  method changes: none")
    lines.append(
        f'  min latency config: {policy_summary["min_latency_config"]}; max latency config: {policy_summary["max_latency_config"]}'
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = build_bert_block_graph(BertBlockConfig(batch_size=1, seq_len=8, hidden_size=768, intermediate_size=3072))
    results: list[dict[str, object]] = []

    for sweep in _network_configs():
        network = NetworkConfig(bandwidth_bytes_per_sec=sweep.bandwidth_bytes_per_sec, rtt_ms=sweep.rtt_ms)
        db = build_planning_profiler_db(graph, network)
        cost_model = CostModel(db=db, default_strategy="exact")
        assignment_result, compiler_plan, runtime_plan = compile_graph_to_runtime_plan(graph, cost_model)

        if assignment_result.assignment.get("n2") != "MPC":
            raise RuntimeError(
                "Planner selected Attention_QK_MatMul outside MPC; current BERT-block IR/capability path requires MPC for n2. "
                f"assignment={assignment_result.assignment}"
            )

        operator_steps = [step for step in runtime_plan.steps if step.type == "operator"]
        nodes = []
        for node, step in zip(graph.nodes, operator_steps):
            nodes.append({
                "node_id": node.node_id,
                "op_type": node.op_type,
                "backend": step.backend.value,
                "method": step.method,
            })

        edges = _conversion_edges(graph, assignment_result.assignment)
        conversion_count = sum(1 for edge in edges if edge["kind"] == "conversion")
        result = {
            "config_name": sweep.name,
            "bandwidth_label": sweep.bandwidth_label,
            "bandwidth_bytes_per_sec": sweep.bandwidth_bytes_per_sec,
            "rtt_ms": sweep.rtt_ms,
            "estimated_total_latency_ms": float(compiler_plan["cost_breakdown"]["total_cost_ms"]),
            "conversion_count": conversion_count,
            "nodes": nodes,
            "edges": edges,
            "simulation_notes": {
                "planning_only": True,
                "attention_support": (
                    "Attention nodes participate through synthetic, shape-compatible planning records. "
                    "This experiment visualizes compiler strategy; it does not claim full real attention execution."
                ),
                "attention_qk_he": (
                    "Attention_QK_MatMul@HE remains structurally invalid for the current BERT-block IR and is kept as a "
                    "high-cost synthetic placeholder so valid plans continue to select MPC."
                ),
            },
        }
        results.append(result)

        slug = _slugify(sweep.name)
        (OUTPUT_DIR / f"{slug}.json").write_text(json.dumps(result, indent=2))
        (OUTPUT_DIR / f"{slug}.dot").write_text(
            _dot_for_strategy(sweep.name, nodes, edges, float(result["estimated_total_latency_ms"]))
        )

    policy_summary = _policy_change_summary(results)
    aggregate = {
        "graph_id": graph.graph_id,
        "node_order": [node.node_id for node in graph.nodes],
        "results": results,
        "policy_summary": policy_summary,
    }
    (OUTPUT_DIR / "aggregate_strategy_sweep.json").write_text(json.dumps(aggregate, indent=2))
    summary_text = _text_summary(results, policy_summary)
    (OUTPUT_DIR / "summary.txt").write_text(summary_text + "\n")
    print(summary_text)
    print(f"aggregate_json={OUTPUT_DIR / 'aggregate_strategy_sweep.json'}")


if __name__ == "__main__":
    main()

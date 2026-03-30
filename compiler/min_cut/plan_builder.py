from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List

from .cost_model import CostModel
from .domain_assignment import OperatorGraph, evaluate_assignment_cost, make_uniform_assignment
from .profiler_db import Domain

NONLINEAR_OPS = {"Softmax", "LayerNorm", "GeLU"}


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    kind: str
    payload: Dict[str, object]


def _topological_order(graph: OperatorGraph) -> List[str]:
    indeg = defaultdict(int)
    outgoing = defaultdict(list)
    for node in graph.nodes:
        indeg[node.node_id] = 0
    for edge in graph.edges:
        indeg[edge.dst] += 1
        outgoing[edge.src].append(edge.dst)
    q = deque([n.node_id for n in graph.nodes if indeg[n.node_id] == 0])
    order: List[str] = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in outgoing[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != len(graph.nodes):
        raise ValueError("Graph contains a cycle; topological order is required for plan building.")
    return order


def build_execution_plan(
    graph: OperatorGraph,
    assignment: Dict[str, Domain],
    cost_model: CostModel,
    include_baselines: bool = True,
) -> Dict[str, object]:
    node_map = {n.node_id: n for n in graph.nodes}
    incoming = defaultdict(list)
    for edge in graph.edges:
        incoming[edge.dst].append(edge)

    steps: List[Dict[str, object]] = []
    step_idx = 1
    order = _topological_order(graph)

    for node_id in order:
        node = node_map[node_id]
        node_domain = assignment[node_id]
        for edge in incoming[node_id]:
            src_domain = assignment[edge.src]
            if src_domain != node_domain:
                conv = cost_model.estimate_conversion_cost(
                    tensor_shape=edge.tensor_shape,
                    from_domain=src_domain,
                    to_domain=node_domain,
                )
                steps.append(
                    {
                        "step_id": f"s{step_idx}",
                        "kind": "conversion",
                        "from_node": edge.src,
                        "to_node": edge.dst,
                        "from_domain": src_domain,
                        "to_domain": node_domain,
                        "tensor_shape": list(edge.tensor_shape),
                        "estimated_latency_ms": conv.latency_ms,
                    }
                )
                step_idx += 1

        op_cost = cost_model.estimate_node_cost(
            op_type=node.op_type,
            domain=node_domain,
            input_shape=node.input_shape,
            output_shape=node.output_shape,
        )
        steps.append(
            {
                "step_id": f"s{step_idx}",
                "kind": "operator",
                "node_id": node.node_id,
                "op_type": node.op_type,
                "domain": node_domain,
                "input_shape": list(node.input_shape),
                "output_shape": list(node.output_shape),
                "estimated_latency_ms": op_cost.latency_ms,
            }
        )
        step_idx += 1

    node_cost_ms, conversion_cost_ms, total_cost_ms = evaluate_assignment_cost(graph, assignment, cost_model)
    plan: Dict[str, object] = {
        "graph_id": graph.graph_id,
        "assignment": assignment,
        "steps": steps,
        "cost_breakdown": {
            "node_cost_ms": node_cost_ms,
            "conversion_cost_ms": conversion_cost_ms,
            "total_cost_ms": total_cost_ms,
        },
    }
    if include_baselines:
        all_he = make_uniform_assignment(graph, "HE")
        all_mpc = make_uniform_assignment(graph, "MPC")
        hybrid_linear_he = {
            node.node_id: ("MPC" if node.op_type in NONLINEAR_OPS else "HE")
            for node in graph.nodes
        }
        _, _, all_he_total = evaluate_assignment_cost(graph, all_he, cost_model)
        _, _, all_mpc_total = evaluate_assignment_cost(graph, all_mpc, cost_model)
        _, _, hybrid_total = evaluate_assignment_cost(graph, hybrid_linear_he, cost_model)
        plan["baselines"] = {
            "all_he_total_ms": all_he_total,
            "all_mpc_total_ms": all_mpc_total,
            "hybrid_linear_he_nonlinear_mpc_total_ms": hybrid_total,
        }
    return plan


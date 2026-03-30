from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from compiler.capability_checker import default_capability_checker
from runtime.plan import ConversionStep, ExecutionPlan, OperatorStep
from runtime.types import BackendType

from backends.layout.bert_edge_packing import supports_bert_edge_conversion_shape
from .cost_model import CostModel
from .domain_assignment import AssignmentResult, OperatorGraph, assign_domains_min_cut
from .plan_builder import build_execution_plan
from .profiler_db import Domain


DOMAIN_METHOD_DEFAULTS: Dict[Domain, str] = {
    "HE": "method_he_nexus",
    "MPC": "method_mpc_bolt",
}

DOMAIN_METHOD_OVERRIDES: Dict[tuple[str, Domain], str] = {
    ("Embedding", "HE"): "method_runtime_default",
    ("Embedding", "MPC"): "method_runtime_default",
    ("Linear_QKV", "HE"): "method_runtime_default",
    ("Linear_QKV", "MPC"): "method_runtime_default",
    ("Out_Projection", "HE"): "method_runtime_default",
    ("Out_Projection", "MPC"): "method_runtime_default",
    ("Residual_Add", "HE"): "method_runtime_default",
    ("Residual_Add", "MPC"): "method_runtime_default",
    ("Attention_QK_MatMul", "MPC"): "method_mpc",
    ("Attention_V_MatMul", "MPC"): "method_mpc",
}


CONVERSION_METHOD_DEFAULT = "method_default"
CONVERSION_METHOD_SCI_RESTRICTED = "method_sci_restricted"


def resolve_conversion_method(
    from_op_type: str,
    to_op_type: str,
    tensor_shape: tuple[int, ...],
) -> str:
    if supports_bert_edge_conversion_shape(tensor_shape):
        return CONVERSION_METHOD_SCI_RESTRICTED
    return CONVERSION_METHOD_DEFAULT


def _backend(domain: Domain) -> BackendType:
    return BackendType(domain)


def _incoming_edge_map(graph: OperatorGraph):
    incoming = defaultdict(list)
    for edge in graph.edges:
        incoming[edge.dst].append(edge)
    return incoming


def _source_nodes(graph: OperatorGraph) -> List[str]:
    incoming = {edge.dst for edge in graph.edges}
    return [node.node_id for node in graph.nodes if node.node_id not in incoming]


def resolve_method_name(op_type: str, domain: Domain) -> str:
    if domain not in DOMAIN_METHOD_DEFAULTS:
        raise KeyError(f"Missing compiler-side domain method default for domain={domain}")
    return DOMAIN_METHOD_OVERRIDES.get((op_type, domain), DOMAIN_METHOD_DEFAULTS[domain])


def _node_attributes(node: object) -> Dict[str, object]:
    return dict(getattr(node, "attributes", {}) or {})


def _resolve_validated_method(node: object, domain: Domain) -> str:
    op_type = str(getattr(node, "op_type"))
    input_shape = tuple(getattr(node, "input_shape"))
    attributes = _node_attributes(node)
    method_name = resolve_method_name(op_type, domain)
    if not default_capability_checker.is_method_valid(op_type, method_name, input_shape, attributes):
        raise ValueError(
            "Compiler-side method selection failed contract validation: "
            f"op={op_type}, domain={domain}, method={method_name}, "
            f"input_shape={list(input_shape)}, attributes={attributes}"
        )
    return method_name


def compiler_plan_to_runtime_plan(
    graph: OperatorGraph,
    compiler_plan: Dict[str, object],
    external_inputs: Dict[str, List[str]] | None = None,
) -> ExecutionPlan:
    node_map = {node.node_id: node for node in graph.nodes}
    incoming = _incoming_edge_map(graph)
    provided_inputs = dict(external_inputs or {})
    source_nodes = _source_nodes(graph)
    for node_id in source_nodes:
        provided_inputs.setdefault(node_id, ["input"])

    runtime_steps: List[OperatorStep | ConversionStep] = []
    node_output_tensor: Dict[str, str] = {}
    conversion_output_tensor: Dict[tuple[str, str], str] = {}

    for raw_step in compiler_plan["steps"]:  # type: ignore[index]
        step = dict(raw_step)
        kind = str(step["kind"])
        if kind == "conversion":
            from_node = str(step["from_node"])
            to_node = str(step["to_node"])
            src_domain = str(step["from_domain"])
            dst_domain = str(step["to_domain"])
            source_tensor = node_output_tensor.get(from_node, f"{from_node}__out")
            output_tensor = f"{from_node}__to__{to_node}"
            runtime_steps.append(
                ConversionStep(
                    from_domain=_backend(src_domain),  # type: ignore[arg-type]
                    to_domain=_backend(dst_domain),  # type: ignore[arg-type]
                    tensor=source_tensor,
                    method=resolve_conversion_method(
                        str(getattr(node_map[from_node], "op_type")),
                        str(getattr(node_map[to_node], "op_type")),
                        tuple(step.get("tensor_shape", [])),
                    ),
                    output_tensor=output_tensor,
                )
            )
            conversion_output_tensor[(from_node, to_node)] = output_tensor
            continue

        if kind != "operator":
            raise ValueError(f"Unsupported compiler plan step kind: {kind}")

        node_id = str(step["node_id"])
        op_type = str(step["op_type"])
        domain = str(step["domain"])
        node = node_map[node_id]
        input_tensor_names: List[str] = []
        if incoming[node_id]:
            for edge in incoming[node_id]:
                input_tensor_names.append(
                    conversion_output_tensor.get((edge.src, node_id), node_output_tensor.get(edge.src, f"{edge.src}__out"))
                )
        else:
            if node_id not in provided_inputs:
                raise KeyError(f"Missing external input mapping for source node {node_id}")
            input_tensor_names.extend(provided_inputs[node_id])

        output_tensor = f"{node_id}__out"
        runtime_steps.append(
            OperatorStep(
                op_type=op_type,
                method=_resolve_validated_method(node, domain),  # type: ignore[arg-type]
                backend=_backend(domain),  # type: ignore[arg-type]
                inputs=input_tensor_names,
                outputs=[output_tensor],
            )
        )
        node_output_tensor[node_id] = output_tensor

    return ExecutionPlan(steps=runtime_steps)


def compile_graph_to_runtime_plan(
    graph: OperatorGraph,
    cost_model: CostModel,
    external_inputs: Dict[str, List[str]] | None = None,
    include_baselines: bool = True,
) -> tuple[AssignmentResult, Dict[str, object], ExecutionPlan]:
    assignment_result = assign_domains_min_cut(graph, cost_model)
    compiler_plan = build_execution_plan(
        graph,
        assignment_result.assignment,
        cost_model,
        include_baselines=include_baselines,
    )
    runtime_plan = compiler_plan_to_runtime_plan(
        graph,
        compiler_plan,
        external_inputs=external_inputs,
    )
    return assignment_result, compiler_plan, runtime_plan

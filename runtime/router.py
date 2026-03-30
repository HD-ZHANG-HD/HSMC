from __future__ import annotations

import warnings
import json
from pathlib import Path
from typing import Dict, List

from .executor import execute
from .operator_registry import OperatorRegistry
from .operator_specs import BERT_OPERATOR_SEQUENCE
from .plan import ConversionStep, ExecutionPlan, OperatorStep
from .types import BackendType, ExecutionContext, TensorValue


class DomainConverters:
    @staticmethod
    def he_to_mpc(tensor: TensorValue, ctx: ExecutionContext) -> TensorValue:
        raise RuntimeError("DomainConverters is deprecated. Use explicit conversion steps in an ExecutionPlan.")

    @staticmethod
    def mpc_to_he(tensor: TensorValue, ctx: ExecutionContext) -> TensorValue:
        raise RuntimeError("DomainConverters is deprecated. Use explicit conversion steps in an ExecutionPlan.")

    @staticmethod
    def to_hybrid(tensor: TensorValue, ctx: ExecutionContext) -> TensorValue:
        raise RuntimeError("DomainConverters is deprecated. Use explicit conversion steps in an ExecutionPlan.")

    @classmethod
    def convert(cls, tensor: TensorValue, target: BackendType, ctx: ExecutionContext) -> TensorValue:
        raise RuntimeError("DomainConverters is deprecated. Use explicit conversion steps in an ExecutionPlan.")


class OperatorRouter:
    def __init__(self, registry: OperatorRegistry, op_backend_map: Dict[str, BackendType]) -> None:
        self.registry = registry
        self.op_backend_map = op_backend_map

    @classmethod
    def from_config_file(cls, registry: OperatorRegistry, path: str | Path) -> "OperatorRouter":
        cfg = json.loads(Path(path).read_text())
        typed = {op: BackendType(v) for op, v in cfg.items()}
        return cls(registry=registry, op_backend_map=typed)

    def build_legacy_plan(self, tensors: Dict[str, TensorValue]) -> ExecutionPlan:
        warnings.warn(
            "OperatorRouter is deprecated. Build an explicit ExecutionPlan and call runtime.execute(...).",
            DeprecationWarning,
            stacklevel=2,
        )
        steps: List[OperatorStep | ConversionStep] = []
        for spec in BERT_OPERATOR_SEQUENCE:
            backend = self.op_backend_map.get(spec.name, BackendType.MPC)
            planned_inputs: List[str] = []
            for input_name in spec.input_names:
                tensor_name = input_name
                tensor = tensors[tensor_name]
                if tensor.domain != backend:
                    converted_name = f"{tensor_name}__to_{backend.value.lower()}__for_{spec.name.lower()}"
                    steps.append(
                        ConversionStep(
                            from_domain=tensor.domain,
                            to_domain=backend,
                            tensor=tensor_name,
                            output_tensor=converted_name,
                        )
                    )
                    tensor_name = converted_name
                    tensors[tensor_name] = TensorValue(tensor.data, backend, dict(tensor.meta))
                planned_inputs.append(tensor_name)
            steps.append(
                OperatorStep(
                    op_type=spec.name,
                    backend=backend,
                    method="method_default",
                    inputs=planned_inputs,
                    outputs=[spec.output_name],
                )
            )
            tensors[spec.output_name] = TensorValue(None, backend)
        return ExecutionPlan(steps)

    def execute_pipeline(self, tensors: Dict[str, TensorValue], ctx: ExecutionContext | None = None) -> Dict[str, TensorValue]:
        ctx = ctx or ExecutionContext()
        plan = self.build_legacy_plan(dict(tensors))
        return execute(plan, dict(tensors), ctx=ctx, registry=self.registry)


def execute_legacy(
    registry: OperatorRegistry,
    op_backend_map: Dict[str, BackendType],
    tensors: Dict[str, TensorValue],
    ctx: ExecutionContext | None = None,
) -> Dict[str, TensorValue]:
    router = OperatorRouter(registry=registry, op_backend_map=op_backend_map)
    return router.execute_pipeline(tensors, ctx=ctx)

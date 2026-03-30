from __future__ import annotations

from ..types import BackendType, ExecutionContext, TensorValue
from .registry import ConversionRegistry, conversion_registry


class ConversionManager:
    def __init__(self, registry: ConversionRegistry | None = None) -> None:
        self.registry = registry or conversion_registry

    def resolve_method_name(
        self,
        src_domain: BackendType,
        dst_domain: BackendType,
        ctx: ExecutionContext,
        method_name: str | None = None,
    ) -> str:
        if method_name:
            return method_name
        direction_key = f"{src_domain.value.lower()}_to_{dst_domain.value.lower()}_method"
        return str(ctx.params.get(direction_key, "method_default"))

    def convert(
        self,
        tensor: TensorValue,
        target: BackendType,
        ctx: ExecutionContext,
        method_name: str | None = None,
    ) -> TensorValue:
        if tensor.domain == target:
            return tensor

        resolved_method = self.resolve_method_name(tensor.domain, target, ctx, method_name)
        method = self.registry.get(tensor.domain, target, resolved_method)
        converted = method(tensor, ctx)
        ctx.trace.append(
            "CONVERT "
            f"{tensor.domain.value}_to_{target.value}"
            f"@{resolved_method}"
            f"[{method.spec.status.value}]"
        )
        return converted


conversion_manager = ConversionManager()

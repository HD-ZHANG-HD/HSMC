from __future__ import annotations

from typing import Dict

from .base import ConversionMethod
from .capability import conversion_capability_registry
from .types import ConversionKey, ConversionMethodSpec


class ConversionRegistry:
    def __init__(self) -> None:
        self._methods: Dict[ConversionKey, ConversionMethod] = {}

    def register(self, spec: ConversionMethodSpec) -> None:
        method = ConversionMethod(spec)
        self._methods[spec.key] = method
        conversion_capability_registry.set_status(
            spec.src_domain,
            spec.dst_domain,
            spec.method_name,
            spec.status,
        )

    def get(self, src_domain, dst_domain, method_name: str) -> ConversionMethod:
        key = ConversionKey(src_domain, dst_domain, method_name)
        if key not in self._methods:
            raise KeyError(
                "Missing conversion implementation: "
                f"{src_domain.value}->{dst_domain.value} via {method_name}"
            )
        return self._methods[key]


conversion_registry = ConversionRegistry()

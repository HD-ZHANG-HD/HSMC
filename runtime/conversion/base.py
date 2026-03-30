from __future__ import annotations

from dataclasses import dataclass

from ..types import ExecutionContext, TensorValue
from .types import ConversionFn, ConversionMethodSpec


@dataclass(frozen=True)
class ConversionMethod:
    spec: ConversionMethodSpec

    def __call__(self, tensor: TensorValue, ctx: ExecutionContext) -> TensorValue:
        return self.spec.fn(tensor, ctx)

    @property
    def fn(self) -> ConversionFn:
        return self.spec.fn

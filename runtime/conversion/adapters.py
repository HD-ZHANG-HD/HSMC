
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ..types import ExecutionContext, TensorValue


@dataclass(frozen=True)
class ConversionContract:
    direction: str
    tensor_shape: tuple[int, ...]
    layout_family: str
    layout_name: str
    ring_bits: int
    scale_bits: int
    assumptions: tuple[str, ...]
    unsupported_cases: tuple[str, ...]

    def as_meta(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "tensor_shape": list(self.tensor_shape),
            "layout_family": self.layout_family,
            "layout_name": self.layout_name,
            "ring_bits": self.ring_bits,
            "scale_bits": self.scale_bits,
            "assumptions": list(self.assumptions),
            "unsupported_cases": list(self.unsupported_cases),
        }


class HeToMpcAdapter(Protocol):
    def convert(self, tensor: TensorValue, meta: ConversionContract, ctx: ExecutionContext) -> TensorValue:
        ...


class MpcToHeAdapter(Protocol):
    def convert(self, tensor: TensorValue, meta: ConversionContract, ctx: ExecutionContext) -> TensorValue:
        ...

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..types import BackendType, ExecutionContext, TensorValue


ConversionFn = Callable[[TensorValue, ExecutionContext], TensorValue]


@dataclass(frozen=True)
class ConversionKey:
    src_domain: BackendType
    dst_domain: BackendType
    method_name: str


@dataclass(frozen=True)
class ConversionMethodSpec:
    src_domain: BackendType
    dst_domain: BackendType
    method_name: str
    status: str
    fn: ConversionFn

    @property
    def key(self) -> ConversionKey:
        return ConversionKey(self.src_domain, self.dst_domain, self.method_name)

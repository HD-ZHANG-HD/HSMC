from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

from .types import BackendType, ExecutionContext, TensorValue


OperatorFn = Callable[[List[TensorValue], ExecutionContext], TensorValue]


@dataclass(frozen=True)
class OperatorKey:
    op_name: str
    backend: BackendType
    method_name: str


class OperatorRegistry:
    def __init__(self) -> None:
        self._impls: Dict[OperatorKey, OperatorFn] = {}
        self._default_methods: Dict[tuple[str, BackendType], str] = {}

    def register(
        self,
        op_name: str,
        backend: BackendType,
        fn: OperatorFn,
        method_name: str | None = None,
    ) -> None:
        resolved_method = method_name or "method_default"
        key = OperatorKey(op_name=op_name, backend=backend, method_name=resolved_method)
        self._impls[key] = fn
        self._default_methods.setdefault((op_name, backend), resolved_method)

    def get(
        self,
        op_name: str,
        backend: BackendType,
        method_name: str | None = None,
    ) -> OperatorFn:
        resolved_method = method_name or self._default_methods.get((op_name, backend), "method_default")
        key = OperatorKey(op_name=op_name, backend=backend, method_name=resolved_method)
        if key not in self._impls:
            raise KeyError(
                "Missing implementation: "
                f"op={op_name}, backend={backend.value}, method={resolved_method}"
            )
        return self._impls[key]


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OperatorProfileRecord:
    op_type: str
    backend: str
    method: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    local_compute_ms: float
    comm_bytes: int
    comm_rounds: int
    total_latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_type": self.op_type,
            "backend": self.backend,
            "domain": self.backend,
            "method": self.method,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "local_compute_ms": self.local_compute_ms,
            "comm_bytes": self.comm_bytes,
            "comm_rounds": self.comm_rounds,
            "total_latency_ms": self.total_latency_ms,
            "latency_ms": self.total_latency_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ConversionProfileRecord:
    direction: str
    method: str
    layout_family: str
    tensor_shape: tuple[int, ...]
    local_compute_ms: float
    comm_bytes: int
    comm_rounds: int
    total_latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from_domain, to_domain = self.direction.split("_to_")
        return {
            "direction": self.direction,
            "from_domain": from_domain,
            "to_domain": to_domain,
            "method": self.method,
            "layout_family": self.layout_family,
            "tensor_shape": list(self.tensor_shape),
            "local_compute_ms": self.local_compute_ms,
            "comm_bytes": self.comm_bytes,
            "comm_rounds": self.comm_rounds,
            "total_latency_ms": self.total_latency_ms,
            "latency_ms": self.total_latency_ms,
            "metadata": dict(self.metadata),
        }

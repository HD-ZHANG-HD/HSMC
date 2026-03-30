from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class BackendType(str, Enum):
    MPC = "MPC"
    HE = "HE"
    HYBRID = "HYBRID"


@dataclass
class TensorValue:
    data: Any
    domain: BackendType
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    params: Dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
    profiling_collector: Any | None = None
    network_config: Any | None = None


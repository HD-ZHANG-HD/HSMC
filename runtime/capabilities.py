from __future__ import annotations

from enum import Enum
from typing import Dict

from .operator_specs import BERT_OPERATOR_SEQUENCE
from .types import BackendType


class CapabilityStatus(str, Enum):
    REAL_INTEGRATED = "real-integrated"
    RESTRICTED_INTEGRATED = "restricted-integrated"
    MOCK = "mock"
    UNSUPPORTED = "unsupported"


class BackendCapabilityRegistry:
    def __init__(self) -> None:
        self._status: Dict[str, Dict[BackendType, CapabilityStatus]] = {}
        for spec in BERT_OPERATOR_SEQUENCE:
            self._status[spec.name] = {
                BackendType.MPC: CapabilityStatus.MOCK,
                BackendType.HE: CapabilityStatus.MOCK,
                BackendType.HYBRID: CapabilityStatus.MOCK,
            }

    def set_status(self, op_name: str, backend: BackendType, status: CapabilityStatus) -> None:
        if op_name not in self._status:
            self._status[op_name] = {}
        self._status[op_name][backend] = status

    def get_status(self, op_name: str, backend: BackendType) -> CapabilityStatus:
        return self._status.get(op_name, {}).get(backend, CapabilityStatus.UNSUPPORTED)

    def snapshot(self) -> Dict[str, Dict[str, str]]:
        return {
            op: {backend.value: status.value for backend, status in backend_map.items()}
            for op, backend_map in self._status.items()
        }


capability_registry = BackendCapabilityRegistry()
capability_registry.set_status("GeLU", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("Softmax", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("LayerNorm", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("FFN_Linear_1", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("Attention_QK_MatMul", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("Attention_V_MatMul", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("Residual_Add", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("FFN_Linear_2", BackendType.MPC, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("GeLU", BackendType.HE, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("Softmax", BackendType.HE, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("Residual_Add", BackendType.HE, CapabilityStatus.REAL_INTEGRATED)
capability_registry.set_status("LayerNorm", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)
capability_registry.set_status("FFN_Linear_1", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)
capability_registry.set_status("FFN_Linear_2", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)
capability_registry.set_status("Attention_QK_MatMul", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)
capability_registry.set_status("Attention_V_MatMul", BackendType.HE, CapabilityStatus.RESTRICTED_INTEGRATED)

from __future__ import annotations

from typing import Dict

from ..capabilities import CapabilityStatus
from ..types import BackendType


class ConversionCapabilityRegistry:
    def __init__(self) -> None:
        self._status: Dict[tuple[BackendType, BackendType], Dict[str, CapabilityStatus]] = {}

    def set_status(
        self,
        src_domain: BackendType,
        dst_domain: BackendType,
        method_name: str,
        status: CapabilityStatus,
    ) -> None:
        key = (src_domain, dst_domain)
        self._status.setdefault(key, {})[method_name] = status

    def get_status(
        self,
        src_domain: BackendType,
        dst_domain: BackendType,
        method_name: str,
    ) -> CapabilityStatus:
        key = (src_domain, dst_domain)
        return self._status.get(key, {}).get(method_name, CapabilityStatus.UNSUPPORTED)

    def snapshot(self) -> Dict[str, Dict[str, str]]:
        return {
            f"{src.value}_to_{dst.value}": {
                method_name: status.value for method_name, status in method_map.items()
            }
            for (src, dst), method_map in self._status.items()
        }


conversion_capability_registry = ConversionCapabilityRegistry()

from __future__ import annotations

from ...capabilities import CapabilityStatus
from ...types import BackendType, ExecutionContext, TensorValue
from ..types import ConversionMethodSpec


METHOD_NAME = "method_default"


def convert_mpc_to_he(tensor: TensorValue, ctx: ExecutionContext) -> TensorValue:
    del ctx
    return TensorValue(tensor.data, BackendType.HE, dict(tensor.meta))


METHOD_SPEC = ConversionMethodSpec(
    src_domain=BackendType.MPC,
    dst_domain=BackendType.HE,
    method_name=METHOD_NAME,
    status=CapabilityStatus.MOCK,
    fn=convert_mpc_to_he,
)

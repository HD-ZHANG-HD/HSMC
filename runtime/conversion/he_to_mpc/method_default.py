from __future__ import annotations

from ...capabilities import CapabilityStatus
from ...types import BackendType, ExecutionContext, TensorValue
from ..types import ConversionMethodSpec


METHOD_NAME = "method_default"


def convert_he_to_mpc(tensor: TensorValue, ctx: ExecutionContext) -> TensorValue:
    del ctx
    return TensorValue(tensor.data, BackendType.MPC, dict(tensor.meta))


METHOD_SPEC = ConversionMethodSpec(
    src_domain=BackendType.HE,
    dst_domain=BackendType.MPC,
    method_name=METHOD_NAME,
    status=CapabilityStatus.MOCK,
    fn=convert_he_to_mpc,
)

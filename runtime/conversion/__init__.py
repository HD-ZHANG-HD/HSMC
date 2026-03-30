from .capability import ConversionCapabilityRegistry, conversion_capability_registry
from .manager import ConversionManager, conversion_manager
from .registry import ConversionRegistry, conversion_registry
from .types import ConversionFn, ConversionKey, ConversionMethodSpec

from .he_to_mpc.method_default import METHOD_SPEC as HE_TO_MPC_METHOD_DEFAULT
from .he_to_mpc.method_sci_restricted import METHOD_SPEC as HE_TO_MPC_METHOD_SCI_RESTRICTED
from .mpc_to_he.method_default import METHOD_SPEC as MPC_TO_HE_METHOD_DEFAULT
from .mpc_to_he.method_sci_restricted import METHOD_SPEC as MPC_TO_HE_METHOD_SCI_RESTRICTED

conversion_registry.register(HE_TO_MPC_METHOD_DEFAULT)
conversion_registry.register(HE_TO_MPC_METHOD_SCI_RESTRICTED)
conversion_registry.register(MPC_TO_HE_METHOD_DEFAULT)
conversion_registry.register(MPC_TO_HE_METHOD_SCI_RESTRICTED)

__all__ = [
    "ConversionCapabilityRegistry",
    "ConversionFn",
    "ConversionKey",
    "ConversionManager",
    "ConversionMethodSpec",
    "ConversionRegistry",
    "conversion_capability_registry",
    "conversion_manager",
    "conversion_registry",
]

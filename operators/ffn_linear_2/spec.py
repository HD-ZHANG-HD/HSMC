"""Operator metadata for FFN_Linear_2."""

OPERATOR_NAME = "FFN_Linear_2"

OPERATOR_SPEC = {
    "name": OPERATOR_NAME,
    "inputs": ["ffn_activated"],
    "outputs": ["ffn_out"],
    "attributes": {
        "supports_method_dispatch": True,
        "default_method": "method_mpc_bolt",
        "available_methods": ["method_mpc_bolt", "method_he_nexus"],
        "semantic_lowering": "reuse_existing_linear_execution",
        "he_nexus_status": "restricted-integrated",
    },
}

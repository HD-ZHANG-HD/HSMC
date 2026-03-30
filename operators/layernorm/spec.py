"""Operator metadata for LayerNorm."""

OPERATOR_NAME = "LayerNorm"

OPERATOR_SPEC = {
    "name": OPERATOR_NAME,
    "inputs": ["attn_residual"],
    "outputs": ["attn_norm"],
    "attributes": {
        "supports_method_dispatch": True,
        "default_method": "method_mpc_bolt",
        "available_methods": ["method_mpc_bolt", "method_he_nexus"],
        "he_nexus_status": "restricted-integrated",
    },
}

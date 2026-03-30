"""Operator metadata for GeLU."""

OPERATOR_NAME = "GeLU"

OPERATOR_SPEC = {
    "name": OPERATOR_NAME,
    "inputs": ["ffn_hidden"],
    "outputs": ["ffn_activated"],
    "attributes": {
        "supports_method_dispatch": True,
        "default_method": "method_mpc_bolt",
        "available_methods": ["method_mpc_bolt", "method_he_nexus"],
        "he_nexus_note": "approximate_gelu_emulation",
    },
}


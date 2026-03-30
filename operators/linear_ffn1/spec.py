"""Operator metadata for FFN_Linear_1."""

OPERATOR_NAME = "FFN_Linear_1"

OPERATOR_SPEC = {
    "name": OPERATOR_NAME,
    "inputs": ["attn_norm"],
    "outputs": ["ffn_hidden"],
    "attributes": {
        "supports_method_dispatch": True,
        "default_method": "method_mpc_bolt",
        "configurable_output_dim": True,
        "available_methods": ["method_mpc_bolt", "method_he_nexus"],
        "he_nexus_status": "restricted-integrated",
        "he_nexus_contract": {
            "input_shape": "[B,S,H]",
            "output_shape": "[B,S,64]",
            "restrictions": ["H=768", "1<=B*S<=4096"],
        },
    },
}


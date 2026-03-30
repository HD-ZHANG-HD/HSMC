"""Operator metadata for Softmax."""

OPERATOR_NAME = "Softmax"

OPERATOR_SPEC = {
    "name": OPERATOR_NAME,
    "inputs": ["qk_scores"],
    "outputs": ["attn_probs"],
    "attributes": {
        "supports_method_dispatch": True,
        "default_method": "method_mpc_bolt",
        "available_methods": ["method_mpc_bolt", "method_he_nexus"],
        "he_nexus_note": "approximate_softmax_emulation",
    },
}


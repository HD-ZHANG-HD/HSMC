"""Operator metadata for Attention_QK_MatMul."""

OPERATOR_NAME = "Attention_QK_MatMul"

OPERATOR_SPEC = {
    "name": OPERATOR_NAME,
    "inputs": ["qkv_out"],
    "outputs": ["qk_scores"],
    "attributes": {
        "supports_method_dispatch": True,
        "default_method": "method_mpc",
        "available_methods": ["method_mpc", "method_he_nexus"],
        "supports_packed_qkv_input": True,
        "supports_presplit_qk_input": True,
        "internal_canonical_shape": "[B,H,S,D]",
        "he_nexus_status": "restricted-integrated",
        "he_nexus_contract": {
            "input_layout": "[3,B,S,768]",
            "output_layout": "[B,12,S,S]",
            "restrictions": ["heads=12", "1<=B<=8", "1<=S<=128"],
        },
    },
}


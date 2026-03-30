"""Operator metadata for Attention_V_MatMul."""

OPERATOR_NAME = "Attention_V_MatMul"

OPERATOR_SPEC = {
    "name": OPERATOR_NAME,
    "inputs": ["attn_probs", "qkv_out"],
    "outputs": ["context"],
    "attributes": {
        "supports_method_dispatch": True,
        "default_method": "method_mpc",
        "available_methods": ["method_mpc", "method_he_nexus"],
        "supports_packed_qkv_input": True,
        "supports_presplit_v_input": True,
        "internal_canonical_shape": "[B,H,S,D]",
        "default_output_shape": "[B,S,H*D]",
        "optional_output_flag": "attention_return_canonical",
        "he_nexus_status": "restricted-integrated",
        "he_nexus_contract": {
            "input_layout": ["[B,12,S,S]", "[3,B,S,768]"],
            "output_layout_default": "[B,S,768]",
            "output_layout_canonical": "[B,12,S,64]",
            "restrictions": ["heads=12", "1<=B<=8", "1<=S<=128"],
        },
    },
}


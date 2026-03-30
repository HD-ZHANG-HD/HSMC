"""Operator metadata for Residual_Add."""

OPERATOR_NAME = "Residual_Add"

OPERATOR_SPEC = {
    "name": OPERATOR_NAME,
    "inputs": ["attn_proj", "embedding_out"],
    "outputs": ["attn_residual"],
    "attributes": {
        "supports_method_dispatch": True,
        "default_method": "method_runtime_default",
        "available_methods": ["method_runtime_default"],
        "semantic_lowering": "backend_tensor_add",
    },
}

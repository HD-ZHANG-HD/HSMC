from __future__ import annotations

from compiler.capability_checker import get_valid_methods, is_method_valid


def main() -> None:
    print("[LayerNorm]")
    print(
        "he_nexus valid:",
        is_method_valid("LayerNorm", "method_he_nexus", (1, 8, 768), {}),
    )
    print(
        "he_nexus invalid tokens:",
        is_method_valid("LayerNorm", "method_he_nexus", (2, 16, 768), {}),
    )
    print(
        "valid methods:",
        get_valid_methods("LayerNorm", (1, 8, 768), {}),
    )

    print("[Attention_QK_MatMul]")
    print(
        "he_nexus packed valid:",
        is_method_valid(
            "Attention_QK_MatMul",
            "method_he_nexus",
            (3, 1, 8, 768),
            {"packed_qkv": True, "num_heads": 12},
        ),
    )
    print(
        "he_nexus unpacked invalid:",
        is_method_valid(
            "Attention_QK_MatMul",
            "method_he_nexus",
            (1, 8, 768),
            {"packed_qkv": False},
        ),
    )
    print(
        "valid methods:",
        get_valid_methods("Attention_QK_MatMul", (3, 1, 8, 768), {"packed_qkv": True}),
    )

    print("[FFN]")
    print(
        "ffn1 he_nexus valid:",
        is_method_valid("FFN_Linear_1", "method_he_nexus", (1, 8, 768), {"out_dim": 64}),
    )
    print(
        "ffn1 he_nexus invalid hidden:",
        is_method_valid("FFN_Linear_1", "method_he_nexus", (1, 8, 512), {"out_dim": 64}),
    )
    print(
        "ffn2 he_nexus valid:",
        is_method_valid("FFN_Linear_2", "method_he_nexus", (1, 8, 1536), {"out_dim": 768}),
    )
    print(
        "ffn2 valid methods:",
        get_valid_methods("FFN_Linear_2", (1, 8, 1536), {"out_dim": 768}),
    )


if __name__ == "__main__":
    main()

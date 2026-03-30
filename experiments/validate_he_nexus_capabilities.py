from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from operators.attention_qk_matmul.method_he_nexus import run_nexus_attention_qk_matmul_he
from operators.attention_v_matmul.method_he_nexus import run_nexus_attention_v_matmul_he
from operators.layernorm.method_he_nexus import run_nexus_layernorm_he
from operators.linear_ffn1.method_he_nexus import run_nexus_linear_ffn1_he
from runtime import BackendType, capability_registry


def _expect_unsupported(name: str, fn, args) -> None:
    try:
        fn(*args)
    except NotImplementedError:
        print(f"[method-status] {name}=unsupported")
        return
    raise AssertionError(f"{name}: expected NotImplementedError")


def _expect_value_error(name: str, fn, args) -> None:
    try:
        fn(*args)
    except ValueError:
        print(f"[method-contract] {name}=rejects_unsupported_input")
        return
    raise AssertionError(f"{name}: expected ValueError")


def main() -> None:
    expected = {
        "GeLU": "real-integrated",
        "Softmax": "real-integrated",
        "LayerNorm": "restricted-integrated",
        "FFN_Linear_1": "restricted-integrated",
        "Attention_QK_MatMul": "restricted-integrated",
        "Attention_V_MatMul": "restricted-integrated",
    }

    for op, status in expected.items():
        got = capability_registry.get_status(op, BackendType.HE).value
        assert got == status, f"{op}@HE expected {status}, got {got}"
        print(f"[capability] {op}@HE={got}")

    layernorm = run_nexus_layernorm_he(np.zeros((1, 2, 768)))
    assert layernorm.shape == (1, 2, 768), f"LayerNorm.method_he_nexus: shape mismatch {layernorm.shape}"
    print("[method-status] LayerNorm.method_he_nexus=restricted-integrated")
    _expect_value_error("LayerNorm.method_he_nexus", run_nexus_layernorm_he, (np.zeros((1, 2, 4)),))
    y = run_nexus_linear_ffn1_he(np.zeros((1, 2, 768)))
    assert y.shape == (1, 2, 64), f"FFN_Linear_1.method_he_nexus: shape mismatch {y.shape}"
    print("[method-status] FFN_Linear_1.method_he_nexus=restricted-integrated")
    qk = run_nexus_attention_qk_matmul_he([np.zeros((3, 1, 2, 768))])
    assert qk.shape == (1, 12, 2, 2), f"Attention_QK_MatMul.method_he_nexus shape mismatch {qk.shape}"
    print("[method-status] Attention_QK_MatMul.method_he_nexus=restricted-integrated")

    v = run_nexus_attention_v_matmul_he([np.zeros((1, 12, 2, 2)), np.zeros((3, 1, 2, 768))])
    assert v.shape == (1, 2, 768), f"Attention_V_MatMul.method_he_nexus shape mismatch {v.shape}"
    print("[method-status] Attention_V_MatMul.method_he_nexus=restricted-integrated")


if __name__ == "__main__":
    main()

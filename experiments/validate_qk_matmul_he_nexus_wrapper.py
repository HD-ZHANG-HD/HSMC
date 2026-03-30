from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from operators.attention_qk_matmul.method_he_nexus import (
    NexusHeAttentionQkMatMulConfig,
    run_nexus_attention_qk_matmul_he,
)


def run_supported_case(name: str, b: int, s: int) -> None:
    cfg = NexusHeAttentionQkMatMulConfig()
    qkv = np.random.standard_normal((3, b, s, 768))
    out = run_nexus_attention_qk_matmul_he([qkv], cfg=cfg)
    assert out.shape == (b, 12, s, s), f"{name}: shape mismatch {out.shape}"
    assert np.isfinite(out).all(), f"{name}: non-finite output"
    print(f"[wrapper-test] {name}: shape_ok=True")


def run_restriction_checks() -> None:
    cfg = NexusHeAttentionQkMatMulConfig()
    try:
        run_nexus_attention_qk_matmul_he([np.zeros((1, 2, 768))], cfg=cfg)
    except ValueError:
        print("[wrapper-test] restriction_layout_ok=True")
    else:
        raise AssertionError("Expected restricted layout failure")

    try:
        run_nexus_attention_qk_matmul_he([np.zeros((3, 1, 2, 512))], cfg=cfg)
    except ValueError:
        print("[wrapper-test] restriction_hidden_ok=True")
    else:
        raise AssertionError("Expected restricted hidden-size failure")


def main() -> None:
    run_supported_case("small_sanity", 1, 2)
    run_supported_case("multi_batch", 4, 8)
    run_supported_case("nontrivial_seq", 2, 16)
    run_restriction_checks()


if __name__ == "__main__":
    main()

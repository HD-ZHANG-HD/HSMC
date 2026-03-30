from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from runtime.types import ExecutionContext
from operators.attention_v_matmul.method_he_nexus import (
    NexusHeAttentionVMatMulConfig,
    run_nexus_attention_v_matmul_he,
)


def run_supported_case(name: str, b: int, s: int) -> None:
    cfg = NexusHeAttentionVMatMulConfig()
    qkv = np.random.standard_normal((3, b, s, 768))
    attn = np.random.standard_normal((b, 12, s, s))
    out = run_nexus_attention_v_matmul_he([attn, qkv], cfg=cfg)
    assert out.shape == (b, s, 768), f"{name}: shape mismatch {out.shape}"
    assert np.isfinite(out).all(), f"{name}: non-finite output"
    print(f"[wrapper-test] {name}: shape_ok=True")


def run_restriction_checks() -> None:
    cfg = NexusHeAttentionVMatMulConfig()
    try:
        run_nexus_attention_v_matmul_he([np.zeros((1, 2, 2)), np.zeros((3, 1, 2, 768))], cfg=cfg)
    except ValueError:
        print("[wrapper-test] restriction_attn_shape_ok=True")
    else:
        raise AssertionError("Expected restricted attn shape failure")

    out = run_nexus_attention_v_matmul_he(
        [np.zeros((1, 12, 2, 2)), np.zeros((3, 1, 2, 768))],
        ctx=ExecutionContext(params={"attention_return_canonical": True}),
        cfg=cfg,
    )
    assert out.shape == (1, 12, 2, 64), f"canonical output shape mismatch {out.shape}"
    print("[wrapper-test] canonical_output_ok=True")


def main() -> None:
    run_supported_case("small_sanity", 1, 2)
    run_supported_case("multi_batch", 4, 8)
    run_supported_case("nontrivial_seq", 2, 16)
    run_restriction_checks()


if __name__ == "__main__":
    main()
